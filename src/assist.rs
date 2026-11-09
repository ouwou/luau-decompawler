//! `--assist`: a model names a script's variables, and the compiler decides
//! which of its names stay.
//!
//! Bytecode keeps function names, fields and strings, but no local's name, so
//! the decompiler's names for locals and parameters are its own guesses. A
//! model that reads the whole script guesses better, but it can also be wrong
//! in ways that would change what the script does: a name a global or another
//! variable already has where the variable is used, a field or a table key
//! taken for the variable, a declaration it misread. So the model only
//! proposes, one line per rename (`FILE LINE OLD NEW`, LINE being where the
//! variable is declared); the renaming is ours, by scope; and a script keeps a
//! rename only when it still compiles to the very same bytecode. Compiled at
//! debug level 1 a script carries no local's name at all, so the same bytes
//! are the same program: nothing a rename shadows, captures or breaks gets
//! through.
//!
//! The model may also propose edits: lines it would write otherwise, such as
//! a call to a local function where the decompiler shows the body the
//! compiler inlined. An edit moves lines, so it is kept when the script
//! compiles to the same program but for its line numbers (bytecode::skeleton):
//! the same functions, instructions and constants. A script on one line keeps
//! its line numbers (some check their own), and takes no edits.
//!
//! The model is the `claude` command in print mode, with no tools and a system
//! prompt of our own: scripts go several to a request, several requests at
//! once.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::passes::{
    alnum_len, alpha_len, code_mask, func_open, has_var_word, indent_of, is_blank, is_token, parse_for_in,
    parse_for_num, parse_local_list, pass_keyword, table_key_at,
};

/// The model asked unless another is named, and how hard it thinks.
pub const MODEL: &str = "claude-opus-5-5";
pub const EFFORT: &str = "low";
/// Scripts to a request and requests at once, unless told otherwise. A
/// request takes about 6 s before its first word whatever its size, so
/// requests at once matter most: 24 scripts of about 130 lines took 94 s one
/// by one, 41 s in one request, 12 s four to a request six at once.
pub const BATCH: usize = 4;
pub const JOBS: usize = 6;
/// Lines a request holds before it takes another script: a big script goes
/// alone, and a request's answer stays short enough to come back quickly.
const LINES: usize = 2000;

pub struct Options {
    pub model: String,
    pub effort: String,
    /// Scripts in one request, and requests running at once.
    pub batch: usize,
    pub jobs: usize,
    /// The `claude` command, and luau_check, which checks every rename.
    pub claude: PathBuf,
    pub check: PathBuf,
    /// Edits asked for as well as names, and a comment above functions.
    pub edits: bool,
    pub comments: bool,
    /// A line on standard error as each request is answered.
    pub progress: bool,
}

/// luau_check: the one given, else in $LUAU_BIN, else beside this program.
pub fn find_check(given: Option<&str>) -> Option<PathBuf> {
    let exe = if cfg!(windows) { "luau_check.exe" } else { "luau_check" };
    let mut tries: Vec<PathBuf> = Vec::new();
    if let Some(g) = given {
        tries.push(PathBuf::from(g));
    }
    if let Some(dir) = std::env::var_os("LUAU_BIN") {
        tries.push(Path::new(&dir).join(exe));
    }
    if let Some(dir) = std::env::current_exe().ok().as_deref().and_then(Path::parent) {
        tries.push(dir.join(exe));
    }
    tries.into_iter().find(|p| p.is_file())
}

/// A script to name: what to call it in the request, and its text.
pub struct Script {
    pub name: String,
    pub text: String,
}

/// What became of a script: its new text when any rename or edit stayed.
#[derive(Default)]
pub struct Done {
    pub text: Option<String>,
    pub proposed: usize,
    pub kept: usize,
    pub edits_proposed: usize,
    pub edits_kept: usize,
    pub comments_proposed: usize,
    pub comments_kept: usize,
    pub error: Option<String>,
    /// What was not kept, and why: `LINE OLD -> NEW: why`, `edit FIRST-LAST: why`.
    pub refused: Vec<String>,
}

/// The requests' time and tokens, summed.
#[derive(Default, Clone)]
pub struct Totals {
    pub requests: usize,
    pub failed: usize,
    /// Seconds the requests took, added up (they overlap).
    pub busy: f64,
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

const NAMES: &str = "You restore decompiled Roblox Luau to read like the source it came from.

The bytecode kept function names, field and method names and strings, but not the names of locals and parameters, so the decompiler made those up: placeholders (v5_1, arg2_1, t2, n3, v2x2), generic names (value, flag, callback, list, t, n, s), or names taken from the first value a variable holds. Propose a better name wherever the current one is a placeholder, generic, misleading or inconsistent with the rest of the code. Keep names that already say what the variable is.

Rules for names:
- Rename only local variables, parameters and loop variables. Never rename functions declared with `function` or `local function`, globals, fields, methods, or anything inside a string.
- Be consistent: one thing gets one name everywhere, in every function and in every file of the request; parameters with the same role in sibling functions get the same name; the same module, service or object gets the same name in each file.
- Follow Roblox conventions: PascalCase for services, required modules and classes (Players, ReplicatedStorage, Signal); camelCase for other values (character, humanoid, rootPart, connection). Use whole words.
- When nothing in the code says what a variable is, leave it.

Give each rename as one line:
FILE LINE OLD NEW
FILE is the file's number, LINE the number of the line that declares the variable (its `local`, `for` or `function(...)` line), OLD its name now and NEW the new one. A name declared on several lines is several variables: give a line for each one you rename.";

const EDITS: &str = "

You may also rewrite lines the decompiler wrote less plainly than the source would have. Give each edit as:
EDIT FILE FIRST LAST
the new lines, indented as in the file
END
FIRST and LAST are the first and last line it replaces, as numbered in the request (edits of one file may not overlap); write the new lines with the names you propose. Every edit is compiled with Luau at -O2 and kept only if the compiled program is exactly the same apart from line numbers: the same instructions in the same order, the same constants, the same locals declared in the same places. So rewrite only what compiles to the same code. The clearest case is a call to a local function where the decompiler shows the function's body copied in (Luau inlines small local functions at -O2): the call compiles alike when the variables receiving its results are declared where the call stands, not earlier. Anything else, or anything in another order, does not compile the same and is thrown away. Do not add comments, and leave code that is already clear as it is.

Give every rename as a rename line, those of variables in lines you edit too: an edit that does not compile the same is thrown away, and its lines then keep only the renames.";

const NOTES: &str = "

Above each function whose purpose the code makes clear, add a comment saying in plain words what it does, as one line:
NOTE FILE LINE text
LINE is the line the function starts on; the comment goes above it. One short sentence, and none where you would be guessing.";

const ANSWER: &str = "

Answer with these lines and nothing else.";

/// The system prompt: names, and edits and comments when asked for.
fn system(edits: bool, comments: bool) -> String {
    format!("{NAMES}{}{}{ANSWER}", if edits { EDITS } else { "" }, if comments { NOTES } else { "" })
}

/// The request for a group of scripts: each under its number and name, its
/// lines numbered as the answer refers to them.
pub fn prompt(group: &[&Script]) -> String {
    let mut s = String::new();
    for (k, script) in group.iter().enumerate() {
        s.push_str(&format!("=== FILE {}: {} ===\n", k + 1, script.name));
        for (i, line) in script.text.split('\n').enumerate() {
            s.push_str(&format!("{}| {}\n", i + 1, line));
        }
        s.push('\n');
    }
    s
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rename {
    /// The line declaring the variable, counted from 1.
    pub line: usize,
    pub old: String,
    pub new: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edit {
    /// The lines it replaces, counted from 1, both ends included.
    pub first: usize,
    pub last: usize,
    /// What they become.
    pub lines: Vec<String>,
    /// A comment the model put above a line (`NOTE`), not an edit of code.
    pub note: bool,
}

/// What an answer proposes for one script.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Answer {
    pub renames: Vec<Rename>,
    pub edits: Vec<Edit>,
}

/// What an answer proposes for each of `files` scripts, in its order:
/// `FILE LINE OLD NEW` lines (`OLD -> NEW` too), and `EDIT FILE FIRST LAST`
/// ... `END`; anything else is ignored.
pub fn parse(answer: &str, files: usize) -> Vec<Answer> {
    let mut out = vec![Answer::default(); files];
    let mut lines = answer.lines();
    while let Some(l) = lines.next() {
        let w: Vec<&str> = l.split_whitespace().filter(|w| *w != "->").collect();
        // `NOTE FILE LINE text`: a comment above that line, kept as an edit
        // whose one line is the comment's text until the line is known
        if w.first() == Some(&"NOTE") && w.len() >= 4 {
            let (Ok(file), Ok(line)) = (w[1].parse::<usize>(), w[2].parse::<usize>()) else { continue };
            // the text after the third word
            let mut after = l.trim_start();
            for _ in 0..3 {
                let e = after.find(char::is_whitespace).unwrap_or(after.len());
                after = after[e..].trim_start();
            }
            let after = after.trim_end();
            if (1..=files).contains(&file) && line >= 1 && !after.is_empty() {
                let text: String = after.chars().take(200).collect();
                out[file - 1].edits.push(Edit { first: line, last: line, lines: vec![text], note: true });
            }
            continue;
        }
        if w.first() == Some(&"EDIT") {
            // the new lines, to END, any code fence around them left out
            let mut body: Vec<String> = Vec::new();
            let mut ended = false;
            for b in lines.by_ref() {
                if b.trim() == "END" {
                    ended = true;
                    break;
                }
                body.push(b.trim_end_matches('\r').to_string());
            }
            if body.first().is_some_and(|f| f.trim_start().starts_with("```")) {
                body.remove(0);
            }
            if body.last().is_some_and(|f| f.trim() == "```") {
                body.pop();
            }
            let n: Vec<Option<usize>> = w.iter().skip(1).map(|v| v.parse().ok()).collect();
            if let (true, [Some(file), Some(first), Some(last)]) = (ended, n.as_slice()) {
                if (1..=files).contains(file) && *first >= 1 && last >= first {
                    out[file - 1].edits.push(Edit { first: *first, last: *last, lines: body, note: false });
                }
            }
            continue;
        }
        if w.len() != 4 {
            continue;
        }
        let (Ok(file), Ok(line)) = (w[0].parse::<usize>(), w[1].parse::<usize>()) else { continue };
        if file == 0 || file > files || line == 0 || !is_token(w[2]) || !is_token(w[3]) || w[2] == w[3] {
            continue;
        }
        out[file - 1].renames.push(Rename { line, old: w[2].to_string(), new: w[3].to_string() });
    }
    out
}

/// The lines with these edits made, bottom first so that each one's numbers
/// still hold; None when one reaches past the end or two overlap.
fn edited(base: &[String], edits: &[&Edit]) -> Option<Vec<String>> {
    let mut order: Vec<&Edit> = edits.to_vec();
    order.sort_by_key(|e| std::cmp::Reverse(e.first));
    let mut out = base.to_vec();
    let mut below = usize::MAX;
    for e in order {
        if e.first == 0 || e.last < e.first || e.last > base.len() || e.last >= below {
            return None;
        }
        out.splice(e.first - 1..e.last, e.lines.iter().cloned());
        below = e.first;
    }
    Some(out)
}

// ---- renaming ---------------------------------------------------------------

/// A variable's declaration: its name at byte `at` of the declaring line, and
/// the lines `lo..hi` where that name means it (but where a declaration in
/// there takes the name over).
struct Decl {
    at: usize,
    lo: usize,
    hi: usize,
    /// Every name its list declares (its own too): no two may be the same.
    list: Vec<String>,
}

/// The first line after `d` no deeper than it: the end of the block line `d`
/// opens (its `end`, `end)`, `else`...), or the end of the text.
fn block_close(lines: &[String], d: usize) -> usize {
    let pad = indent_of(&lines[d]);
    (d + 1..lines.len()).find(|&j| !is_blank(&lines[j]) && indent_of(&lines[j]) <= pad).unwrap_or(lines.len())
}

/// Where a local declared at depth `pad` stops being seen, its statement
/// ending before line `from`: the first shallower line, which ends its block
/// (`until` still sees it), or the end of the text.
fn local_hi(lines: &[String], from: usize, pad: usize) -> usize {
    match (from..lines.len()).find(|&j| !is_blank(&lines[j]) && indent_of(&lines[j]) < pad) {
        Some(j) if lines[j].trim_start().starts_with("until") => j + 1,
        Some(j) => j,
        None => lines.len(),
    }
}

/// The comma-separated names of `list`, which starts at byte `base` of its
/// line: the byte where `name` stands among them, a type after it allowed.
/// The names in a comma-separated list, types after them left out.
fn list_names(list: &str) -> Vec<String> {
    list.split(',').map(|p| p.split(':').next().unwrap_or("").trim().to_string()).collect()
}

fn name_in_list(list: &str, base: usize, name: &str) -> Option<usize> {
    let mut pos = base;
    for part in list.split(',') {
        let lead = part.len() - part.trim_start().len();
        let end = part.find(':').unwrap_or(part.len());
        if part[..end].trim() == name {
            return Some(pos + lead);
        }
        pos += part.len() + 1;
    }
    None
}

/// The declaration of `name` on line `d`, if the line declares it: as a
/// parameter of the function the line opens, a local, or a loop variable.
/// A function's own name is not a variable to rename here.
fn declaration(lines: &[String], d: usize, name: &str) -> Result<Option<Decl>, &'static str> {
    let line = &lines[d];
    let pad = indent_of(line);
    let s = &line[pad..];
    let mut found: Vec<Decl> = Vec::new();
    if let Some(r) = s.strip_prefix("local function ") {
        if r.split('(').next().map(str::trim) == Some(name) {
            return Err("a function's name");
        }
    }
    // a parameter: of `function f(...)` or `function(...)` ending the line
    if func_open(line).is_some() {
        let t = line.trim_end();
        let close = t.len() - 1;
        if let Some(open) = t[..close].rfind('(') {
            let params = &t[open + 1..close];
            if let Some(at) = name_in_list(params, open + 1, name) {
                found.push(Decl { at, lo: d + 1, hi: block_close(lines, d), list: list_names(params) });
            }
        }
    }
    if let Some((names, _)) = parse_local_list(s) {
        if let Some(at) = name_in_list(names, pad + 6, name) {
            // a function opened on the line ends the statement at its `end`,
            // and the statement goes on while that line opens another
            let mut last = d;
            while func_open(&lines[last]).is_some() {
                let close = block_close(lines, last);
                if close >= lines.len() || indent_of(&lines[close]) != pad {
                    return Err("a statement that does not end");
                }
                last = close;
            }
            let lo = last + 1;
            found.push(Decl { at, lo, hi: local_hi(lines, lo, pad), list: list_names(names) });
        }
    }
    let loop_vars = parse_for_in(line).map(|(vars, _)| vars).or_else(|| parse_for_num(line));
    if let Some(vars) = loop_vars {
        if let Some(at) = name_in_list(vars, pad + 4, name) {
            found.push(Decl { at, lo: d + 1, hi: block_close(lines, d), list: list_names(vars) });
        }
    }
    match found.len() {
        0 => Ok(None),
        1 => Ok(found.pop()),
        _ => Err("declared twice on its line"),
    }
}

/// Where `name` stands in `line` as a variable: a whole word outside strings,
/// not a field or method after `.` or `:`, not a constructor's key.
fn uses(line: &str, name: &str) -> Vec<usize> {
    let mut out = Vec::new();
    if !line.contains(name) {
        return out;
    }
    let b = line.as_bytes();
    let code = code_mask(line);
    let mut i = 0;
    while i < b.len() {
        if !code[i] {
            i += 1;
            continue;
        }
        let n = alpha_len(line, i);
        if n == 0 {
            i += 1;
            continue;
        }
        let mut j = i + n;
        while j < b.len() {
            let m = alnum_len(line, j);
            if m == 0 {
                break;
            }
            j += m;
        }
        let prev = if i > 0 { b[i - 1] } else { 0 };
        // after `.` a field, but after `..` an operand of the concatenation
        let field = prev == b':' || prev == b'.' && !(i >= 2 && b[i - 2] == b'.');
        if &line[i..j] == name && !field && !table_key_at(line, i, j - i) {
            out.push(i);
        }
        i = j;
    }
    out
}

/// The text with the variable `r.old` declared on line `r.line` renamed, and
/// nothing else; or why not here. The rename is refused where the new name is
/// in use anywhere the variable is seen: the compiler checks the rest.
pub fn rename(lines: &[String], r: &Rename) -> Result<Vec<String>, &'static str> {
    // `local self = setmetatable({}, Class)` is common Lua: self may be given
    // (where a method's own self is used, it is in use there)
    let reserved = r.new != "self" && pass_keyword(&r.new);
    if !is_token(&r.old) || !is_token(&r.new) || reserved || r.new == r.old {
        return Err("not a name to give");
    }
    let given = r.line.checked_sub(1).filter(|d| *d < lines.len()).ok_or("no such line")?;
    // a model counts a line off now and then: the line above or below
    // will do when only one of them declares the name
    let (d, decl) = match declaration(lines, given, &r.old)? {
        Some(decl) => (given, decl),
        None => {
            let near: Vec<(usize, Decl)> = [given.wrapping_sub(1), given + 1]
                .into_iter()
                .filter(|&j| j < lines.len())
                .filter_map(|j| declaration(lines, j, &r.old).ok().flatten().map(|decl| (j, decl)))
                .collect();
            match <[(usize, Decl); 1]>::try_from(near) {
                Ok([one]) => one,
                Err(_) => return Err("not declared on that line"),
            }
        }
    };
    // declarations of the same name in its scope take the name for theirs
    let mut shadows: Vec<(usize, usize)> = Vec::new();
    let mut theirs: Vec<(usize, usize)> = Vec::new();
    for j in decl.lo..decl.hi {
        if let Some(inner) = declaration(lines, j, &r.old)? {
            theirs.push((j, inner.at));
            shadows.push((inner.lo, inner.hi));
        }
    }
    let mut hits: Vec<(usize, usize)> = vec![(d, decl.at)];
    for j in decl.lo..decl.hi {
        if shadows.iter().any(|&(lo, hi)| lo <= j && j < hi) {
            continue;
        }
        for p in uses(&lines[j], &r.old) {
            if !theirs.contains(&(j, p)) {
                hits.push((j, p));
            }
        }
    }
    if decl.list.iter().any(|n| *n == r.new) {
        return Err("the new name is declared with it");
    }
    // the rest of the declaring statement runs before the variable exists:
    // `local require = require` is the global's value under its own name
    if (decl.lo..decl.hi).any(|j| has_var_word(&lines[j], &r.new)) {
        return Err("the new name is in use where the variable is");
    }
    let mut out = lines.to_vec();
    hits.sort_unstable_by(|a, b| b.cmp(a));
    for (j, p) in hits {
        out[j].replace_range(p..p + r.old.len(), &r.new);
    }
    Ok(out)
}

// ---- checking by compiling ----------------------------------------------------

/// Each text compiled by luau_check (-O2, debug level 1): its bytecode, or
/// None where it does not compile.
pub fn compile(check: &Path, texts: &[String]) -> Result<Vec<Option<Vec<u8>>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let mut input = Vec::new();
    for (i, t) in texts.iter().enumerate() {
        input.extend_from_slice(format!("{} {}\n", t.len(), i).as_bytes());
        input.extend_from_slice(t.as_bytes());
    }
    let out = run_piped(&mut Command::new(check), &["--stream".to_string()], input)
        .map_err(|e| format!("{}: {e}", check.display()))?;
    let mut found: Vec<Option<Vec<u8>>> = vec![None; texts.len()];
    let b = &out.0;
    let mut at = 0;
    while let Some(eol) = b[at..].iter().position(|&c| c == b'\n').map(|p| at + p) {
        let head = String::from_utf8_lossy(&b[at..eol]).into_owned();
        let w: Vec<&str> = head.split(' ').collect();
        let (Some(i), Some(len)) = (w.first().and_then(|v| v.parse::<usize>().ok()), w.get(2).and_then(|v| v.parse::<usize>().ok()))
        else {
            break; // the summary line
        };
        if w.len() != 3 || eol + 1 + len > b.len() || i >= texts.len() {
            break;
        }
        if w[1] == "ok" {
            found[i] = Some(b[eol + 1..eol + 1 + len].to_vec());
        }
        at = eol + 1 + len;
    }
    Ok(found)
}

/// A command run with `input` on its standard input, without a window:
/// (standard output, standard error, whether it succeeded).
fn run_piped(cmd: &mut Command, args: &[String], input: Vec<u8>) -> std::io::Result<(Vec<u8>, Vec<u8>, bool)> {
    cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn()?;
    let mut stdin = child.stdin.take().expect("piped");
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let mut out = Vec::new();
    let mut err = Vec::new();
    let mut so = child.stdout.take().expect("piped");
    let mut se = child.stderr.take().expect("piped");
    let reader = std::thread::spawn(move || {
        let _ = se.read_to_end(&mut err);
        err
    });
    so.read_to_end(&mut out)?;
    let err = reader.join().unwrap_or_default();
    let _ = writer.join();
    let status = child.wait()?;
    Ok((out, err, status.success()))
}

/// The renames a script keeps, and its text with them. Each is applied in turn
/// where it can be; if the script then compiles to other bytecode than
/// before, each is tried alone, those that pass together, and failing that
/// one by one.
fn settle(check: &Path, scripts: &[&Script], proposals: &[Vec<Rename>]) -> Vec<Done> {
    let n = scripts.len();
    let bases: Vec<Vec<String>> = scripts.iter().map(|s| s.text.split('\n').map(String::from).collect()).collect();
    let apply = |base: &[String], list: &[Rename]| -> (Vec<String>, Vec<Rename>) {
        let mut cur = base.to_vec();
        let mut ok = Vec::new();
        for r in list {
            if let Ok(next) = rename(&cur, r) {
                cur = next;
                ok.push(r.clone());
            }
        }
        (cur, ok)
    };
    let mut done: Vec<Done> = (0..n).map(|i| Done { proposed: proposals[i].len(), ..Done::default() }).collect();
    let why = |r: &Rename, reason: &str| format!("{} {} -> {}: {reason}", r.line, r.old, r.new);
    let mut applied: Vec<Vec<Rename>> = Vec::with_capacity(n);
    let mut texts: Vec<String> = Vec::with_capacity(2 * n);
    for i in 0..n {
        let mut cur = bases[i].clone();
        let mut ok = Vec::new();
        // in the answer's order, then once more for those whose new name a
        // later rename freed (`a -> b` before `b -> c`)
        let mut later: Vec<&Rename> = Vec::new();
        for r in &proposals[i] {
            match rename(&cur, r) {
                Ok(next) => {
                    cur = next;
                    ok.push(r.clone());
                }
                Err(_) => later.push(r),
            }
        }
        for r in later {
            match rename(&cur, r) {
                Ok(next) => {
                    cur = next;
                    ok.push(r.clone());
                }
                Err(reason) => done[i].refused.push(why(r, reason)),
            }
        }
        texts.push(scripts[i].text.clone());
        texts.push(cur.join("\n"));
        applied.push(ok);
    }
    let first = match compile(check, &texts) {
        Ok(c) => c,
        Err(e) => {
            for d in &mut done {
                d.error = Some(e.clone());
            }
            return done;
        }
    };
    let mut base_code: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
    let mut unsettled: Vec<usize> = Vec::new();
    for i in 0..n {
        let base = first[2 * i].clone();
        if base.is_none() {
            done[i].error = Some("the script does not compile as it is".to_string());
        } else if !applied[i].is_empty() {
            if first[2 * i + 1] == base {
                done[i].kept = applied[i].len();
                done[i].text = Some(texts[2 * i + 1].clone());
            } else {
                unsettled.push(i);
            }
        }
        base_code.push(base);
    }
    if unsettled.is_empty() {
        return done;
    }
    // each rename alone
    let mut singles: Vec<(usize, Rename)> = Vec::new();
    let mut single_texts: Vec<String> = Vec::new();
    for &i in &unsettled {
        for r in &applied[i] {
            if let Ok(lines) = rename(&bases[i], r) {
                singles.push((i, r.clone()));
                single_texts.push(lines.join("\n"));
            }
        }
    }
    let alone = compile(check, &single_texts).unwrap_or_default();
    let mut passed: Vec<Vec<Rename>> = vec![Vec::new(); n];
    for (k, (i, r)) in singles.into_iter().enumerate() {
        if alone.get(k).is_some_and(|c| c.is_some() && *c == base_code[i]) {
            passed[i].push(r);
        } else {
            done[i].refused.push(why(&r, "it changes the bytecode"));
        }
    }
    // those together, else one by one
    for &i in &unsettled {
        let (lines, ok) = apply(&bases[i], &passed[i]);
        if ok.is_empty() {
            continue;
        }
        let text = lines.join("\n");
        let same = compile(check, std::slice::from_ref(&text)).ok().and_then(|mut c| c.pop()).flatten();
        if same.is_some() && same == base_code[i] {
            done[i].kept = ok.len();
            done[i].text = Some(text);
            continue;
        }
        let mut cur = bases[i].clone();
        let mut kept = 0;
        for r in &ok {
            let Ok(next) = rename(&cur, r) else { continue };
            let t = next.join("\n");
            let c = compile(check, std::slice::from_ref(&t)).ok().and_then(|mut c| c.pop()).flatten();
            if c.is_some() && c == base_code[i] {
                cur = next;
                kept += 1;
            }
        }
        if kept > 0 {
            done[i].kept = kept;
            done[i].text = Some(cur.join("\n"));
        }
    }
    done
}

/// Each text compiled and read back as the program it is, line numbers aside
/// (bytecode::skeleton); None where it does not compile.
fn skeletons(check: &Path, texts: &[String]) -> Vec<Option<String>> {
    let compiled = compile(check, texts).unwrap_or_default();
    (0..texts.len())
        .map(|i| {
            let bytes = compiled.get(i)?.as_ref()?;
            let m = crate::bytecode::parse_as(bytes, crate::bytecode::Opcodes::Stock).ok()?;
            Some(crate::bytecode::skeleton(&m))
        })
        .collect()
}

/// The edits each script keeps, made on the text its renames left. An edit
/// stays when the script then compiles to the same program but for its line
/// numbers: all at once if they can, else each alone, those that pass
/// together, and failing that one by one from the bottom.
fn settle_edits(check: &Path, scripts: &[&Script], done: &mut [Done], edits: &[Vec<Edit>]) {
    let n = scripts.len();
    let mut bases: Vec<Vec<String>> = Vec::with_capacity(n);
    // each script's edits, code first (a comment in the way of one goes), the
    // comments made whole lines: the comment, then the line it stands above
    // as the renames left it
    let mut owned: Vec<Vec<Edit>> = Vec::with_capacity(n);
    for i in 0..n {
        let text = done[i].text.clone().unwrap_or_else(|| scripts[i].text.clone());
        let lines: Vec<String> = text.split('\n').map(String::from).collect();
        let mut list: Vec<Edit> = edits[i].iter().filter(|e| !e.note).cloned().collect();
        for e in edits[i].iter().filter(|e| e.note) {
            let mut e = e.clone();
            if let (Some(line), Some(text)) = (lines.get(e.first - 1), e.lines.first()) {
                let pad = &line[..indent_of(line)];
                e.lines = vec![format!("{pad}-- {text}"), line.clone()];
            }
            list.push(e);
        }
        done[i].edits_proposed = list.iter().filter(|e| !e.note).count();
        done[i].comments_proposed = list.iter().filter(|e| e.note).count();
        bases.push(lines);
        owned.push(list);
    }
    let mut wanted: Vec<Vec<&Edit>> = Vec::with_capacity(n);
    for i in 0..n {
        let lines = &bases[i];
        // a script all on one line (blank lines above it keeping the line it
        // was on) checks its own line numbers, some do: they stay
        let one_line = lines.iter().filter(|l| !l.trim().is_empty()).count() <= 1;
        let mut ok: Vec<&Edit> = Vec::new();
        for e in &owned[i] {
            let why = if one_line {
                Some("a script on one line keeps its line numbers")
            } else if e.last > lines.len() {
                Some("it reaches past the end")
            } else if ok.iter().any(|o| o.first <= e.last && e.first <= o.last) {
                Some("it overlaps another edit")
            } else if e.lines[..] == lines[e.first - 1..e.last] {
                Some("it changes nothing")
            } else {
                None
            };
            match why {
                Some(w) => done[i].refused.push(format!("edit {}-{}: {w}", e.first, e.last)),
                None => ok.push(e),
            }
        }
        wanted.push(ok);
    }
    // edits and comments kept, counted apart
    let keep = |d: &mut Done, list: &[&Edit]| {
        d.edits_kept = list.iter().filter(|e| !e.note).count();
        d.comments_kept = list.iter().filter(|e| e.note).count();
    };
    let changes = |i: usize, w: &str, list: &[&Edit], done: &mut [Done]| {
        for e in list {
            done[i].refused.push(format!("edit {}-{}: {w}", e.first, e.last));
        }
    };
    // all of each script's edits at once
    let who: Vec<usize> = (0..n).filter(|&i| !wanted[i].is_empty()).collect();
    let mut texts: Vec<String> = Vec::with_capacity(2 * who.len());
    for &i in &who {
        texts.push(bases[i].join("\n"));
        texts.push(edited(&bases[i], &wanted[i]).unwrap_or_else(|| bases[i].clone()).join("\n"));
    }
    let shapes = skeletons(check, &texts);
    let mut base_shape: Vec<Option<String>> = vec![None; n];
    let mut unsettled: Vec<usize> = Vec::new();
    for (k, &i) in who.iter().enumerate() {
        base_shape[i] = shapes.get(2 * k).cloned().flatten();
        if base_shape[i].is_none() {
            changes(i, "the script does not compile as it is", &wanted[i], done);
        } else if shapes.get(2 * k + 1).cloned().flatten() == base_shape[i] {
            keep(&mut done[i], &wanted[i]);
            done[i].text = Some(texts[2 * k + 1].clone());
        } else {
            unsettled.push(i);
        }
    }
    if unsettled.is_empty() {
        return;
    }
    // each alone
    let mut singles: Vec<(usize, &Edit)> = Vec::new();
    let mut single_texts: Vec<String> = Vec::new();
    for &i in &unsettled {
        for e in &wanted[i] {
            if let Some(lines) = edited(&bases[i], &[e]) {
                singles.push((i, e));
                single_texts.push(lines.join("\n"));
            }
        }
    }
    let alone = skeletons(check, &single_texts);
    let mut passed: Vec<Vec<&Edit>> = vec![Vec::new(); n];
    for (k, (i, e)) in singles.into_iter().enumerate() {
        if alone.get(k).cloned().flatten().is_some_and(|s| Some(s) == base_shape[i]) {
            passed[i].push(e);
        } else {
            changes(i, "it changes the program", &[e], done);
        }
    }
    // those together, else one by one from the bottom (numbers above hold)
    for &i in &unsettled {
        if passed[i].is_empty() {
            continue;
        }
        let together = edited(&bases[i], &passed[i]).map(|l| l.join("\n"));
        if let Some(text) = together {
            if skeletons(check, std::slice::from_ref(&text)).pop().flatten() == base_shape[i] {
                keep(&mut done[i], &passed[i]);
                done[i].text = Some(text);
                continue;
            }
        }
        let mut order = passed[i].clone();
        order.sort_by_key(|e| std::cmp::Reverse(e.first));
        let mut cur = bases[i].clone();
        let mut kept: Vec<&Edit> = Vec::new();
        for e in order {
            let Some(next) = edited(&cur, &[e]) else { continue };
            let t = next.join("\n");
            if skeletons(check, std::slice::from_ref(&t)).pop().flatten() == base_shape[i] {
                cur = next;
                kept.push(e);
            } else {
                changes(i, "it changes the program with the others", &[e], done);
            }
        }
        if !kept.is_empty() {
            keep(&mut done[i], &kept);
            done[i].text = Some(cur.join("\n"));
        }
    }
}

// ---- the model -------------------------------------------------------------------

struct Reply {
    text: String,
    seconds: f64,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

/// One request: the answer's text, its time and its tokens. The command's output is
/// read as `--output-format json` gives it, else as the answer itself.
fn ask(o: &Options, request: &str) -> Result<Reply, String> {
    let t = Instant::now();
    let system = system(o.edits, o.comments);
    let args: Vec<String> = [
        "-p", "--model", &o.model, "--effort", &o.effort, "--tools", "", "--safe-mode", "--strict-mcp-config",
        "--no-session-persistence", "--disable-slash-commands", "--output-format", "json", "--system-prompt", &system,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut cmd = Command::new(&o.claude);
    // no project's settings or memory where it runs
    cmd.current_dir(std::env::temp_dir());
    let (out, err, ok) =
        run_piped(&mut cmd, &args, request.as_bytes().to_vec()).map_err(|e| format!("{}: {e}", o.claude.display()))?;
    let seconds = t.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&out).into_owned();
    let Some(j) = Json::parse(stdout.trim()).and_then(Json::result) else {
        if !ok {
            let why = String::from_utf8_lossy(&err);
            return Err(format!("the model failed: {}", why.trim().lines().last().unwrap_or("no reason given")));
        }
        return Ok(Reply { text: stdout, seconds, input: 0, output: 0, cache_read: 0, cache_write: 0 });
    };
    if j.get("is_error").and_then(Json::boolean) == Some(true) {
        let why = j.get("result").and_then(Json::string).unwrap_or("no reason given");
        return Err(format!("the model failed: {why}"));
    }
    let usage = |k: &str| j.get("usage").and_then(|u| u.get(k)).and_then(Json::number).unwrap_or(0.0) as u64;
    Ok(Reply {
        text: j.get("result").and_then(Json::string).unwrap_or("").to_string(),
        seconds,
        input: usage("input_tokens"),
        output: usage("output_tokens"),
        cache_read: usage("cache_read_input_tokens"),
        cache_write: usage("cache_creation_input_tokens"),
    })
}

/// Names for every script, `o.batch` to a request and `o.jobs` requests at
/// once: what became of each, and the requests' time and tokens.
pub fn run(scripts: &[Script], o: &Options) -> (Vec<Done>, Totals) {
    // scripts in their order (a folder's together, named alike), up to
    // `batch` of them and LINES lines to a request; the biggest go first
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut lines = 0;
    for (i, s) in scripts.iter().enumerate() {
        let n = s.text.lines().count().max(1);
        match batches.last_mut() {
            Some(b) if b.len() < o.batch.max(1) && lines + n <= LINES => b.push(i),
            _ => {
                batches.push(vec![i]);
                lines = 0;
            }
        }
        lines += n;
    }
    let size = |b: &Vec<usize>| b.iter().map(|&i| scripts[i].text.len()).sum::<usize>();
    batches.sort_by_key(|b| std::cmp::Reverse(size(b)));
    let next = AtomicUsize::new(0);
    let done: Vec<Mutex<Done>> = (0..scripts.len()).map(|_| Mutex::new(Done::default())).collect();
    let totals = Mutex::new(Totals::default());
    let workers = o.jobs.clamp(1, batches.len().max(1));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                crate::alloc::adopt();
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    let Some(ids) = batches.get(b) else { break };
                    let group: Vec<&Script> = ids.iter().map(|&i| &scripts[i]).collect();
                    let request = prompt(&group);
                    // one more try after a failure: a request can time out
                    let reply = ask(o, &request).or_else(|_| ask(o, &request));
                    let results = match &reply {
                        Ok(r) => {
                            let answers = parse(&r.text, group.len());
                            let renames: Vec<Vec<Rename>> = answers.iter().map(|a| a.renames.clone()).collect();
                            let mut done = settle(&o.check, &group, &renames);
                            if o.edits || o.comments {
                                let wanted = |e: &Edit| if e.note { o.comments } else { o.edits };
                                let edits: Vec<Vec<Edit>> =
                                    answers.into_iter().map(|a| a.edits.into_iter().filter(wanted).collect()).collect();
                                settle_edits(&o.check, &group, &mut done, &edits);
                            }
                            done
                        }
                        Err(e) => (0..group.len()).map(|_| Done { error: Some(e.clone()), ..Done::default() }).collect(),
                    };
                    {
                        let mut t = totals.lock().unwrap();
                        t.requests += 1;
                        match &reply {
                            Ok(r) => {
                                t.busy += r.seconds;
                                t.input += r.input;
                                t.output += r.output;
                                t.cache_read += r.cache_read;
                                t.cache_write += r.cache_write;
                            }
                            Err(_) => t.failed += 1,
                        }
                    }
                    if o.progress {
                        let kept: usize = results.iter().map(|d| d.kept).sum();
                        let proposed: usize = results.iter().map(|d| d.proposed).sum();
                        match &reply {
                            Ok(r) => eprintln!(
                                "assist: request {} of {}: {} scripts, {kept} of {proposed} renames kept, {:.1}s",
                                b + 1,
                                batches.len(),
                                group.len(),
                                r.seconds
                            ),
                            Err(e) => eprintln!("assist: request {} of {}: {e}", b + 1, batches.len()),
                        }
                    }
                    for (k, d) in results.into_iter().enumerate() {
                        *done[ids[k]].lock().unwrap() = d;
                    }
                }
                crate::alloc::park();
            });
        }
    });
    let totals = totals.into_inner().unwrap();
    (done.into_iter().map(|d| d.into_inner().unwrap()).collect(), totals)
}

// ---- just enough JSON --------------------------------------------------------------

enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(s: &str) -> Option<Json> {
        let mut p = Parser { b: s.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        (p.i == p.b.len()).then_some(v)
    }

    /// `verbose` in the user's settings makes the output every event, the result last.
    /// None when there is no result event: read as output that is not usable JSON.
    fn result(self) -> Option<Json> {
        match self {
            Json::Arr(items) => items.into_iter().rev().find(|e| e.get("type").and_then(Json::string) == Some("result")),
            j => Some(j),
        }
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn string(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    fn number(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    fn boolean(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Json> {
        self.ws();
        match *self.b.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut fields = Vec::new();
                if self.eat(b'}') {
                    return Some(Json::Obj(fields));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    if !self.eat(b':') {
                        return None;
                    }
                    fields.push((k, self.value()?));
                    if self.eat(b'}') {
                        return Some(Json::Obj(fields));
                    }
                    if !self.eat(b',') {
                        return None;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                if self.eat(b']') {
                    return Some(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    if self.eat(b']') {
                        return Some(Json::Arr(items));
                    }
                    if !self.eat(b',') {
                        return None;
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            b't' => self.word(b"true", Json::Bool(true)),
            b'f' => self.word(b"false", Json::Bool(false)),
            b'n' => self.word(b"null", Json::Null),
            _ => {
                let start = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.b[start..self.i]).ok()?.parse().ok().map(Json::Num)
            }
        }
    }

    fn word(&mut self, w: &[u8], v: Json) -> Option<Json> {
        if self.b[self.i..].starts_with(w) {
            self.i += w.len();
            Some(v)
        } else {
            None
        }
    }

    fn string(&mut self) -> Option<String> {
        if self.b.get(self.i) != Some(&b'"') {
            return None;
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let low = self.hex4()?;
                                code = 0x10000 + ((code - 0xD800) << 10) + (low.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            let ch = char::from_u32(code).unwrap_or('\u{FFFD}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                _ => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = std::str::from_utf8(self.b.get(self.i..self.i + 4)?).ok()?;
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Vec<String> {
        s.split('\n').map(String::from).collect()
    }

    fn renamed(text: &str, line: usize, old: &str, new: &str) -> Result<String, &'static str> {
        rename(&lines(text), &Rename { line, old: old.to_string(), new: new.to_string() }).map(|l| l.join("\n"))
    }

    #[test]
    fn a_local_is_renamed_where_it_is_seen_and_nowhere_else() {
        let text = "local value = 1\nlocal t = { value = value, [value] = 2 }\nprint(t.value, value, \"value\", `{value}`)\nlocal value = value + 1\nprint(value)";
        let got = renamed(text, 1, "value", "count").unwrap();
        assert_eq!(
            got,
            "local count = 1\nlocal t = { value = count, [count] = 2 }\nprint(t.value, count, \"value\", `{count}`)\nlocal value = count + 1\nprint(value)"
        );
    }

    #[test]
    fn a_parameter_is_renamed_in_its_function_only() {
        let text = "local function f(a, n)\n    return n + a\nend\nprint(n)\nlocal g = function(n)\n    return n\nend";
        assert_eq!(renamed(text, 1, "n", "count").unwrap(), "local function f(a, count)\n    return count + a\nend\nprint(n)\nlocal g = function(n)\n    return n\nend");
        assert_eq!(renamed(text, 5, "n", "x").unwrap(), "local function f(a, n)\n    return n + a\nend\nprint(n)\nlocal g = function(x)\n    return x\nend");
    }

    #[test]
    fn a_loop_variable_and_an_inner_function_are_scoped() {
        let text = "for _, v in list do\n    task.spawn(function(v)\n        print(v)\n    end, v)\n    print(v)\nend\nprint(v)";
        assert_eq!(
            renamed(text, 1, "v", "item").unwrap(),
            "for _, item in list do\n    task.spawn(function(v)\n        print(v)\n    end, item)\n    print(item)\nend\nprint(v)"
        );
    }

    #[test]
    fn a_local_declared_with_a_function_is_not_seen_inside_it() {
        // `local x = function() return x end`: the inner x is not the local
        let text = "local x = function()\n    return x\nend\nprint(x)";
        assert_eq!(renamed(text, 1, "x", "getX").unwrap(), "local getX = function()\n    return x\nend\nprint(getX)");
    }

    #[test]
    fn a_local_in_a_branch_ends_with_the_branch_and_until_sees_one() {
        let text = "if a then\n    local n = 1\n    print(n)\nelse\n    print(n)\nend\nrepeat\n    local n = f()\nuntil n > 3";
        assert_eq!(renamed(text, 2, "n", "first").unwrap(), "if a then\n    local first = 1\n    print(first)\nelse\n    print(n)\nend\nrepeat\n    local n = f()\nuntil n > 3");
        assert_eq!(renamed(text, 8, "n", "tries").unwrap(), "if a then\n    local n = 1\n    print(n)\nelse\n    print(n)\nend\nrepeat\n    local tries = f()\nuntil tries > 3");
    }

    #[test]
    fn a_line_off_by_one_finds_the_one_declaration_next_to_it() {
        let text = "x:Connect(function(arg1_1)\n    print(arg1_1.Name)\nend)";
        assert_eq!(renamed(text, 2, "arg1_1", "tool").unwrap(), "x:Connect(function(tool)\n    print(tool.Name)\nend)");
        // two declarations next to it: which one is meant is not known
        let two = "local a = 1\nprint(a)\nlocal a = 2\nprint(a)";
        assert!(renamed(two, 2, "a", "b").is_err());
    }

    #[test]
    fn a_name_in_use_where_the_variable_is_is_refused() {
        let text = "local value = 1\nprint(value, count)";
        assert!(renamed(text, 1, "value", "count").is_err()); // a global it would hide
        assert!(renamed(text, 1, "value", "print").is_err());
        assert!(renamed(text, 1, "value", "end").is_err());
        // not declared there, nor next to it
        assert!(renamed("local value = 1\nprint(value)\nprint(value)", 3, "value", "x").is_err());
        assert!(renamed("local function f()\nend", 1, "f", "g").is_err()); // a function's name
    }

    #[test]
    fn the_declaring_statement_runs_before_the_variable_exists() {
        // the right side is the global, read before the local is made
        assert_eq!(
            renamed("local callback = require\nprint(callback(x))", 1, "callback", "require").unwrap(),
            "local require = require\nprint(require(x))"
        );
        // but two names declared together cannot become one
        assert!(renamed("local a, b = 1, 2\nprint(a, b)", 1, "a", "b").is_err());
        assert!(renamed("local function f(a, b)\n    return a\nend", 1, "a", "b").is_err());
        // a constructor's own table may be called self
        assert_eq!(renamed("local t = setmetatable({}, C)\nreturn t", 1, "t", "self").unwrap(), "local self = setmetatable({}, C)\nreturn self");
    }

    #[test]
    fn a_name_freed_by_a_later_rename_is_taken_on_a_second_pass() {
        let Some(check) = find_check(None) else { return };
        let script = Script { name: "s".into(), text: "local a = tostring(1)\nlocal b = tostring(2)\nprint(a, b)".into() };
        let proposals = vec![vec![
            Rename { line: 1, old: "a".into(), new: "b".into() },
            Rename { line: 2, old: "b".into(), new: "c".into() },
        ]];
        let done = settle(&check, &[&script], &proposals);
        assert_eq!(done[0].kept, 2);
        assert_eq!(done[0].text.as_deref(), Some("local b = tostring(1)\nlocal c = tostring(2)\nprint(b, c)"));
    }

    #[test]
    fn answers_are_read_line_by_line() {
        let got = parse("```\n1 3 value healthBar\n2 10 n -> count\nnot a rename\n3 1 a b\n1 0 x y\n```", 2);
        assert_eq!(got[0].renames, vec![Rename { line: 3, old: "value".into(), new: "healthBar".into() }]);
        assert_eq!(got[1].renames, vec![Rename { line: 10, old: "n".into(), new: "count".into() }]);
    }

    #[test]
    fn edits_are_read_to_their_end() {
        let got = parse("1 2 a b\nEDIT 1 4 5\n```lua\n    local x = f()\n\n```\nEND\nEDIT 2 3 1\nnope\nEND\nEDIT 1 9 9\nnever ended", 2);
        assert_eq!(got[0].renames.len(), 1);
        assert_eq!(got[0].edits, vec![Edit { first: 4, last: 5, lines: vec!["    local x = f()".into(), String::new()], note: false }]);
        assert!(got[1].edits.is_empty()); // its range runs backwards
    }

    #[test]
    fn notes_are_comments_above_a_line() {
        let got = parse("NOTE 1 1 Builds the zone table.\nNOTE 1 7   Spaces stay inside  \nNOTE 2 3\nNOTE x 1 no", 1);
        assert_eq!(
            got[0].edits,
            vec![
                Edit { first: 1, last: 1, lines: vec!["Builds the zone table.".into()], note: true },
                Edit { first: 7, last: 7, lines: vec!["Spaces stay inside".into()], note: true },
            ]
        );
    }

    #[test]
    fn edits_are_made_from_the_bottom_and_may_not_overlap() {
        let base = lines("a\nb\nc\nd");
        let e1 = Edit { first: 1, last: 1, lines: vec!["A".into(), "A2".into()], note: false };
        let e2 = Edit { first: 3, last: 4, lines: vec!["CD".into()], note: false };
        assert_eq!(edited(&base, &[&e1, &e2]).unwrap().join("|"), "A|A2|b|CD");
        let e3 = Edit { first: 2, last: 3, lines: vec![], note: false };
        assert!(edited(&base, &[&e2, &e3]).is_none());
        assert!(edited(&base, &[&Edit { first: 4, last: 5, lines: vec![], note: false }]).is_none());
    }

    /// Edits are judged by the program the script compiles to, line numbers
    /// aside: a line more is fine, another value is not.
    #[test]
    fn edits_that_change_the_program_are_not_kept() {
        let Some(check) = find_check(None) else { return };
        let text = "local function double(x)\n    return x * 2\nend\nlocal a = tonumber(\"3\")\nlocal b = a * 2\nprint(b, double(a))";
        let script = Script { name: "s".into(), text: text.into() };
        let edits = vec![vec![
            Edit { first: 4, last: 4, lines: vec!["".into(), "local a = tonumber(\"3\")".into()], note: false },
            Edit { first: 6, last: 6, lines: vec!["print(b, double(a)) -- both".into()], note: false },
            Edit { first: 2, last: 2, lines: vec!["    return x * 3".into()], note: false },
            Edit { first: 5, last: 6, lines: vec!["local b = a * 2".into()], note: false }, // overlaps
        ]];
        let mut done = vec![Done::default()];
        settle_edits(&check, &[&script], &mut done, &edits);
        assert_eq!(done[0].edits_proposed, 4);
        assert_eq!(done[0].edits_kept, 2);
        assert_eq!(
            done[0].text.as_deref(),
            Some("local function double(x)\n    return x * 2\nend\n\nlocal a = tonumber(\"3\")\nlocal b = a * 2\nprint(b, double(a)) -- both")
        );
        // the body Luau copied in, written as the call again
        let inline = Edit { first: 5, last: 5, lines: vec!["local b = double(a)".into()], note: false };
        let mut done = vec![Done::default()];
        settle_edits(&check, &[&script], &mut done, &[vec![inline]]);
        assert_eq!(done[0].edits_kept, 1);
        // a script on one line keeps its lines, blank lines above it or not
        let one = Script { name: "o".into(), text: "\n\nlocal a = 1 print(a)\n".into() };
        let mut done = vec![Done::default()];
        settle_edits(&check, &[&one], &mut done, &[vec![Edit { first: 1, last: 1, lines: vec!["local a = 1".into(), "print(a)".into()], note: false }]]);
        assert_eq!(done[0].edits_kept, 0);
    }

    #[test]
    fn json_of_a_result_is_read() {
        let j = Json::parse(r#"{"result":"1 2 a b\né","is_error":false,"usage":{"input_tokens":12},"duration_ms":6083}"#).unwrap();
        assert_eq!(j.get("result").and_then(Json::string), Some("1 2 a b\né"));
        assert_eq!(j.get("usage").and_then(|u| u.get("input_tokens")).and_then(Json::number), Some(12.0));
        assert_eq!(j.get("is_error").and_then(Json::boolean), Some(false));
        // a lone result is its own result
        let r = Json::parse(r#"{"type":"result","result":"x"}"#).and_then(Json::result).unwrap();
        assert_eq!(r.get("result").and_then(Json::string), Some("x"));
        // verbose: every event, nested and empty arrays in them, the result last
        let events = r#"[{"type":"system","tools":[]},{"type":"assistant","message":{"content":[{"type":"text","text":"1 2 a b"}]}},{"type":"result","result":"1 2 a b","is_error":false,"usage":{"input_tokens":7}}]"#;
        let r = Json::parse(events).and_then(Json::result).unwrap();
        assert_eq!(r.get("result").and_then(Json::string), Some("1 2 a b"));
        assert_eq!(r.get("usage").and_then(|u| u.get("input_tokens")).and_then(Json::number), Some(7.0));
        assert_eq!(r.get("is_error").and_then(Json::boolean), Some(false));
        // no result event: not an answer
        assert!(Json::parse(r#"[{"type":"system"},[]]"#).and_then(Json::result).is_none());
    }

    /// With luau_check at hand ($LUAU_BIN): what a rename would change in the
    /// program is refused even where the text-level checks let it through.
    #[test]
    fn renames_that_change_the_bytecode_are_not_kept() {
        let Some(check) = find_check(None) else { return };
        // the renamer does not know `[[...]]` strings: renaming `value`
        // would change the string too, and the bytecode says so
        let text = "local value = tostring(1)\nlocal callback = function()\n    return [[value]]..value\nend\nlocal function show(n)\n    print(n, callback())\nend\nshow(2)";
        let script = Script { name: "s".into(), text: text.into() };
        let proposals = vec![vec![
            Rename { line: 1, old: "value".into(), new: "label".into() },
            Rename { line: 2, old: "callback".into(), new: "getLabel".into() },
            Rename { line: 5, old: "show".into(), new: "display".into() }, // a function's name
            Rename { line: 5, old: "n".into(), new: "count".into() },
        ]];
        let done = settle(&check, &[&script], &proposals);
        assert_eq!(done[0].proposed, 4);
        assert_eq!(done[0].kept, 2);
        assert_eq!(
            done[0].text.as_deref(),
            Some("local value = tostring(1)\nlocal getLabel = function()\n    return [[value]]..value\nend\nlocal function show(count)\n    print(count, getLabel())\nend\nshow(2)")
        );
    }
}
