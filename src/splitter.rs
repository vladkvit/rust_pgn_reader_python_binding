//! Whole-game boundary detection over a byte buffer.
//!
//! Built directly on the tokenizer: each game is located by running
//! [`tokenizer::parse_game`] with a null visitor and recording where it
//! stopped. Splitter boundaries and parser game-termination semantics can
//! therefore never drift — they are the same code path, not two agreeing
//! implementations. Detection is lexical-only (no SAN parsing, no chess
//! semantics), so it is cheap relative to full parsing.
//!
//! This is the seam a streaming reader needs: given a fixed-size read
//! chunk, the splitter yields the spans of every *complete* game in it and
//! the offset where the trailing partial game starts, so the caller can
//! carry that partial tail into the next chunk.

use crate::tokenizer;
use std::ops::Range;

/// A visitor with all-default (no-op) callbacks: boundary detection needs
/// positions only, not content.
struct NullVisitor;
impl tokenizer::Visitor for NullVisitor {}

/// Find the byte span of every complete game in `buffer`.
///
/// Returns `(spans, rest_start)`:
/// - `spans` — byte ranges of complete games, in order. For every span,
///   running [`tokenizer::parse_game`] on exactly those bytes consumes the
///   whole span (see the round-trip tests).
/// - `rest_start` — offset where the trailing *partial* game begins
///   (`buffer.len()` when the buffer ends at a game boundary or contains
///   only trailing whitespace). `buffer[rest_start..]` is the unparseable
///   remainder to carry over when more bytes are read; it is empty at
///   the end of a complete file.
///
/// Inter-game content is expected to be whitespace. Any other stray
/// content (e.g. previous game lacking a result token) is treated as part
/// of the following game, matching tokenizer game semantics.
pub fn split_games(buffer: &[u8]) -> (Vec<Range<usize>>, usize) {
    let mut spans = Vec::new();
    let mut cursor = 0;
    let len = buffer.len();

    let rest_start = loop {
        // Skip inter-game whitespace.
        while cursor < len && buffer[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= len {
            break len;
        }
        let start = cursor;
        let end = tokenizer::parse_game(&buffer[start..], &mut NullVisitor);
        if !end.terminated {
            // Ran out of input mid-game: partial trailing game.
            break start;
        }
        spans.push(start..start + end.consumed);
        cursor = start + end.consumed;
    };

    (spans, rest_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans_of(buffer: &str) -> Vec<&str> {
        let (spans, _) = split_games(buffer.as_bytes());
        spans.iter().map(|r| &buffer[r.clone()]).collect()
    }

    /// Round-trip agreement: for every emitted span, the tokenizer consumes
    /// the whole span — never stops early (which would mean the splitter's
    /// boundary and the parser's game termination disagree), and never
    /// reports the span as truncated.
    fn assert_round_trip(buffer: &str) {
        let (spans, rest_start) = split_games(buffer.as_bytes());
        for span in &spans {
            let bytes = &buffer.as_bytes()[span.clone()];
            let end = tokenizer::parse_game(bytes, &mut NullVisitor);
            assert_eq!(
                end.consumed,
                bytes.len(),
                "span {:?} not consumed fully: {:?} consumed",
                String::from_utf8_lossy(bytes),
                end
            );
        }
        // Everything after the last span is whitespace or partial-game.
        let tail = &buffer.as_bytes()[rest_start..];
        if rest_start == buffer.len() {
            assert!(tail.is_empty());
        }
    }

    #[test]
    fn empty_and_whitespace_only() {
        let (spans, rest) = split_games(b"");
        assert!(spans.is_empty());
        assert_eq!(rest, 0);

        let ws = b"  \n\t\r\n ";
        let (spans, rest) = split_games(ws);
        assert!(spans.is_empty());
        assert_eq!(rest, ws.len());
    }

    #[test]
    fn two_games_with_blank_line() {
        let buf = "1. e4 e5 1-0\n\n1. d4 d5 0-1\n\n";
        assert_eq!(spans_of(buf), ["1. e4 e5 1-0", "1. d4 d5 0-1"]);
        assert_round_trip(buf);
    }

    #[test]
    fn games_with_headers() {
        let buf = "[Event \"a\"]\n\n1. e4 1-0\n\n[Event \"b\"]\n\n1. d4 0-1";
        assert_eq!(
            spans_of(buf),
            ["[Event \"a\"]\n\n1. e4 1-0", "[Event \"b\"]\n\n1. d4 0-1"]
        );
        assert_round_trip(buf);
    }

    #[test]
    fn game_without_result_separator() {
        // No result between games: the new tag section still terminates.
        // The blank-line terminator's first '\n' is part of the span.
        let buf = "1. e4 e5\n\n[Event \"b\"]\n\n1. d4 *";
        assert_eq!(
            spans_of(buf),
            ["1. e4 e5\n", "[Event \"b\"]\n\n1. d4 *"]
        );
        assert_round_trip(buf);
    }

    #[test]
    fn partial_trailing_game_reported() {
        let buf = "1. e4 1-0\n\n1. d4 d5 2.";
        let (spans, rest) = split_games(buf.as_bytes());
        assert_eq!(spans.len(), 1);
        assert_eq!(&buf[spans[0].clone()], "1. e4 1-0");
        assert_eq!(&buf[rest..], "1. d4 d5 2.");
        assert_round_trip(buf);
    }

    #[test]
    fn unterminated_comment_is_partial() {
        let buf = "1. e4 1-0\n\n1. d4 { never closed";
        let (spans, rest) = split_games(buf.as_bytes());
        assert_eq!(spans.len(), 1);
        assert_eq!(&buf[rest..], "1. d4 { never closed");
        assert_round_trip(buf);
    }

    #[test]
    fn crlf_separators() {
        let buf = "[Event \"a\"]\r\n\r\n1. e4 1-0\r\n\r\n[Event \"b\"]\r\n\r\n1. d4 *\r\n";
        assert_eq!(spans_of(buf).len(), 2);
        assert_round_trip(buf);
    }

    #[test]
    fn headers_only_merge_with_following_game() {
        // A headers-only game directly followed by another header section
        // is ambiguous PGN (no movetext separates the games): per tokenizer
        // semantics the two header sections merge into one game. This is
        // identical to what the one-game-per-slice driver produces for such
        // input, and cannot occur in corpora where every game has movetext.
        let buf = "[Event \"hdr\"]\n[Site \"x\"]\n\n[Event \"g\"]\n\n1. e4 *";
        assert_eq!(spans_of(buf).len(), 1);
        assert_round_trip(buf);
    }

    #[test]
    fn escape_and_comment_content() {
        let buf = "% escaped line\n[Event \"a\"]\n\n1. e4 {has [brackets] and blank?\n\ttext} e5 1-0\n\n1. d4 d5 (1... c5) 0-1";
        assert_eq!(spans_of(buf).len(), 2);
        assert_round_trip(buf);
    }

    #[test]
    fn many_games_all_split() {
        let mut buf = String::new();
        for i in 0..100 {
            if i % 2 == 0 {
                buf.push_str(&format!("[Event \"g{i}\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n\n"));
            } else {
                buf.push_str("1. d4 d5 2. c4 *\n");
            }
        }
        let (spans, rest) = split_games(buf.as_bytes());
        assert_eq!(spans.len(), 100);
        assert_eq!(rest, buf.len());
        assert_round_trip(&buf);
    }
}
