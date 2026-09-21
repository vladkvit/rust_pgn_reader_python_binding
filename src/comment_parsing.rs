//! Hand-rolled single-pass scanner for structured `[%...]` comment tags.
//!
//! Extracts `[%eval ...]` (pawn-unit float, or `#N` mate) and
//! `[%clk H:MM:SS[.fff]]` (remaining clock time, total seconds).
//! Everything else — unknown tags, malformed values, plain text — is
//! ignored: the caller stores the raw comment text verbatim, so nothing is
//! lost by not tokenizing it here.
//!
//! Deliberately lenient (e.g. `[%eval .5]` extracts 0.5, which a strict
//! grammar would reject): for extraction, false negatives lose data while
//! false positives of this shape essentially never occur in real PGN
//! corpora.
//!
//! Replaces the previous nom-based grammar: same extraction results, no
//! dependency, single linear scan.

use memchr::memchr;

/// A structured comment tag extracted by [`scan_tags`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tag {
    /// `[%eval 0.17]` — pawn units.
    Eval(f64),
    /// `[%eval #-3]` — mate in N moves, signed.
    Mate(i32),
    /// `[%clk 0:03:00]` — remaining clock time, total seconds.
    Clk(f64),
}

/// Scan `comment` for structured tags, emitting each via `emit`.
///
/// Tags are `[%-prefixed bracket spans; a span with no closing `]` ends the
/// scan (it cannot contain a complete tag). Unknown and malformed tags are
/// skipped.
pub fn scan_tags(comment: &[u8], mut emit: impl FnMut(Tag)) {
    let mut rest = comment;
    while let Some(open) = memchr(b'[', rest) {
        rest = &rest[open + 1..];
        if rest.first() != Some(&b'%') {
            continue;
        }
        let Some(close) = memchr(b']', rest) else {
            break;
        };
        let body = &rest[1..close];
        let after = &rest[close + 1..];
        if let Some(value) = body.strip_prefix(b"eval") {
            if let Some(tag) = parse_eval_value(value) {
                emit(tag);
            }
        } else if let Some(value) = body.strip_prefix(b"clk") {
            if let Some(tag) = parse_clk_value(value) {
                emit(tag);
            }
        }
        rest = after;
    }
}

/// Parse an eval tag body: a pawn-unit float, or `#N` mate.
fn parse_eval_value(value: &[u8]) -> Option<Tag> {
    let s = std::str::from_utf8(value).ok()?.trim();
    if let Some(mate) = s.strip_prefix('#') {
        return mate.trim_start().parse::<i32>().ok().map(Tag::Mate);
    }
    s.parse::<f64>().ok().map(Tag::Eval)
}

/// Parse a clk tag body: `H:MM:SS` with optional fractional seconds.
fn parse_clk_value(value: &[u8]) -> Option<Tag> {
    let s = std::str::from_utf8(value).ok()?.trim();
    let (hours, rest) = s.split_once(':')?;
    let (minutes, seconds) = rest.split_once(':')?;
    let hours: u32 = hours.parse().ok()?;
    let minutes: u8 = minutes.parse().ok()?;
    let seconds: f64 = seconds.parse().ok()?;
    Some(Tag::Clk(
        hours as f64 * 3600.0 + minutes as f64 * 60.0 + seconds,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(comment: &str) -> Vec<Tag> {
        let mut out = Vec::new();
        scan_tags(comment.as_bytes(), |t| out.push(t));
        out
    }

    #[test]
    fn eval_and_clk_with_text() {
        assert_eq!(
            tags("[%eval 123] some text [%clk 12:34:56]"),
            vec![
                Tag::Eval(123.0),
                Tag::Clk(12.0 * 3600.0 + 34.0 * 60.0 + 56.0)
            ]
        );
    }

    #[test]
    fn clk_fractional_seconds() {
        assert_eq!(
            tags("[%clk 12:34:56.0123]"),
            vec![Tag::Clk(12.0 * 3600.0 + 34.0 * 60.0 + 56.0123)]
        );
        assert_eq!(
            tags("[%clk 1:2:3.45]"),
            vec![Tag::Clk(3600.0 + 2.0 * 60.0 + 3.45)]
        );
    }

    #[test]
    fn eval_mate() {
        assert_eq!(tags("[%eval #-3]"), vec![Tag::Mate(-3)]);
        assert_eq!(tags("[%eval #5]"), vec![Tag::Mate(5)]);
    }

    #[test]
    fn eval_signed_float() {
        assert_eq!(tags("[%eval 123.45]"), vec![Tag::Eval(123.45)]);
        assert_eq!(tags("[%eval -0.5]"), vec![Tag::Eval(-0.5)]);
        assert_eq!(tags("[%eval +3.0]"), vec![Tag::Eval(3.0)]);
    }

    #[test]
    fn malformed_tags_yield_nothing() {
        assert_eq!(tags("[%clk 123]"), vec![]); // clk needs H:M:S
        assert_eq!(tags("[%clk 1::00]"), vec![]);
        assert_eq!(tags("[%clk 1a:00:00]"), vec![]);
        assert_eq!(tags("[%eval notanumber]"), vec![]);
        assert_eq!(tags("[%eval ]"), vec![]);
        assert_eq!(tags("[%eval 0.2 extra]"), vec![]);
    }

    #[test]
    fn unknown_tags_ignored() {
        assert_eq!(tags("[%timestamp 12345] some text"), vec![]);
    }

    #[test]
    fn newline_spacing() {
        assert_eq!(tags("[%clk\n0:09:36.2]"), vec![Tag::Clk(9.0 * 60.0 + 36.2)]);
        assert_eq!(tags("[%eval\n-0.5]"), vec![Tag::Eval(-0.5)]);
    }

    #[test]
    fn brackets_in_text_not_tags() {
        assert_eq!(
            tags("Text with [normal brackets] and then [%clk 1:2:3]"),
            vec![Tag::Clk(3600.0 + 120.0 + 3.0)]
        );
    }

    #[test]
    fn unterminated_tag_ends_scan() {
        assert_eq!(tags("[%eval 0.2"), vec![]);
        // ... but earlier complete tags are still emitted.
        assert_eq!(tags("[%eval 0.2] trailing [%clk 1:2"), vec![Tag::Eval(0.2)]);
    }

    #[test]
    fn empty_and_plain_text() {
        assert_eq!(tags(""), vec![]);
        assert_eq!(tags("just a comment"), vec![]);
    }
}
