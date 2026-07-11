use memchr::memchr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    WhiteWins,
    BlackWins,
    Draw,
    Unknown,
}

/// Purely lexical PGN event consumer.
///
/// The tokenizer performs no chess-semantic work: SAN tokens are delivered
/// as raw bytes via `san_token` and it is the visitor's job to parse them
/// (e.g. with `shakmaty::san::SanPlus::from_ascii`). This keeps a pure
/// counting pass free of SAN-parsing cost.
pub trait Visitor {
    fn begin_headers(&mut self);
    fn header(&mut self, key: &[u8], value: &[u8]);
    fn end_headers(&mut self);
    /// A mainline movetext token that is not a move number, NAG, result or
    /// comment. Usually a SAN move, but syntactically invalid tokens are
    /// delivered raw as well.
    fn san_token(&mut self, token: &[u8]);
    fn comment(&mut self, comment: &[u8]);
    fn outcome(&mut self, outcome: Outcome);
    fn end_game(&mut self);
}

/// Parse a single PGN game from `bytes`, dispatching events to `visitor`.
///
/// Contract: one game per input slice. Content after the first game (a blank
/// line followed by anything, or a new tag section) is ignored, matching the
/// behavior of a single `pgn_reader::Reader::read_game` call.
///
/// Known intentional divergences from pgn-reader:
/// - SAN tokens are not parsed here; they are delivered raw via `san_token`
///   (including syntactically invalid ones, which pgn-reader silently skips —
///   the visitor decides how to handle them).
/// - An unterminated `{` comment is delivered as a comment spanning the rest
///   of the input instead of raising a hard error.
pub fn parse_game<V: Visitor>(mut bytes: &[u8], visitor: &mut V) {
    // Strip UTF-8 BOM
    if bytes.starts_with(b"\xEF\xBB\xBF") {
        bytes = &bytes[3..];
    }

    let mut in_headers = true;
    visitor.begin_headers();

    let mut cursor = 0;
    let len = bytes.len();
    let mut variation_depth = 0usize;

    while cursor < len {
        let c = bytes[cursor];

        match c {
            // '.', '!' and '?' are token delimiters skipped like whitespace;
            // move-number dots ("1.e4") and attached annotations ("e4!?")
            // must not become part of the adjacent token.
            b' ' | b'\r' | b'\t' | b'.' | b'!' | b'?' => {
                cursor += 1;
            }
            b'\n' => {
                cursor += 1;
                match bytes.get(cursor).copied() {
                    // PGN escape: '%' at the start of a line skips the line.
                    Some(b'%') => {
                        // Leave the trailing '\n' so line-start logic reapplies.
                        cursor = memchr(b'\n', &bytes[cursor..]).map_or(len, |o| cursor + o);
                    }
                    // Blank line or a new tag section terminates the movetext
                    // (parity with pgn_reader::Reader::read_game).
                    Some(b'\n') | Some(b'[') if !in_headers => break,
                    Some(b'\r') if !in_headers && bytes.get(cursor + 1) == Some(&b'\n') => break,
                    _ => {}
                }
            }
            // Escape line at the very start of the input.
            b'%' if cursor == 0 => {
                cursor = memchr(b'\n', bytes).unwrap_or(len);
            }
            b'[' => {
                if !in_headers {
                    // Start of a new game but we are already in movetext -> ignore trailing games
                    break;
                }
                cursor += 1;

                // Parse key
                let key_start = cursor;
                while cursor < len
                    && bytes[cursor] != b' '
                    && bytes[cursor] != b'"'
                    && bytes[cursor] != b']'
                {
                    cursor += 1;
                }
                let key = &bytes[key_start..cursor];

                // Find quote
                while cursor < len && bytes[cursor] != b'"' {
                    cursor += 1;
                }

                if cursor < len {
                    cursor += 1; // skip quote
                    let val_start = cursor;
                    // Find closing quote, handling escapes
                    let mut val_end = cursor;
                    while cursor < len {
                        if bytes[cursor] == b'\\' && cursor + 1 < len {
                            cursor += 2;
                        } else if bytes[cursor] == b'"' {
                            val_end = cursor;
                            cursor += 1;
                            break;
                        } else {
                            cursor += 1;
                        }
                    }

                    visitor.header(key, &bytes[val_start..val_end]);
                }

                // Find closing bracket
                while cursor < len && bytes[cursor] != b']' {
                    cursor += 1;
                }
                if cursor < len {
                    cursor += 1;
                }
            }
            b'{' => {
                if in_headers {
                    in_headers = false;
                    visitor.end_headers();
                }
                cursor += 1;
                let comment_start = cursor;
                if let Some(end_offset) = memchr(b'}', &bytes[cursor..]) {
                    if variation_depth == 0 {
                        visitor.comment(&bytes[comment_start..cursor + end_offset]);
                    }
                    cursor += end_offset + 1;
                } else {
                    // No closing brace, consume rest of file
                    if variation_depth == 0 {
                        visitor.comment(&bytes[comment_start..]);
                    }
                    cursor = len;
                }
            }
            b';' => {
                // Line comment
                if in_headers {
                    in_headers = false;
                    visitor.end_headers();
                }
                // Leave the trailing '\n' so line-start logic reapplies.
                cursor = memchr(b'\n', &bytes[cursor..]).map_or(len, |o| cursor + o);
            }
            b'(' => {
                if in_headers {
                    in_headers = false;
                    visitor.end_headers();
                }
                variation_depth += 1;
                cursor += 1;
            }
            b')' => {
                if variation_depth > 0 {
                    variation_depth -= 1;
                }
                cursor += 1;
            }
            b'$' => {
                if in_headers {
                    in_headers = false;
                    visitor.end_headers();
                }
                cursor += 1;
                while cursor < len && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }
            }
            b'*' => {
                if variation_depth == 0 {
                    if in_headers {
                        in_headers = false;
                        visitor.end_headers();
                    }
                    visitor.outcome(Outcome::Unknown);
                    break; // End of game
                }
                cursor += 1;
            }
            _ => {
                // Read a token
                if in_headers {
                    in_headers = false;
                    visitor.end_headers();
                }
                let token_start = cursor;
                while cursor < len && !is_token_delimiter(bytes[cursor]) {
                    cursor += 1;
                }
                let token = &bytes[token_start..cursor];

                if token.is_empty() {
                    // Stray delimiter without a dedicated arm (']' or '}').
                    cursor += 1;
                    continue;
                }

                if variation_depth > 0 {
                    continue; // Skip everything inside variations
                }

                // Check for outcomes and move numbers
                if token == b"1-0" {
                    visitor.outcome(Outcome::WhiteWins);
                    break;
                } else if token == b"0-1" {
                    visitor.outcome(Outcome::BlackWins);
                    break;
                } else if token == b"1/2-1/2" {
                    visitor.outcome(Outcome::Draw);
                    break;
                } else if token.iter().all(|b| b.is_ascii_digit()) {
                    // Move number, skip
                } else {
                    // SAN (or garbage) token — delivered raw, parsed by the visitor.
                    visitor.san_token(token);
                }
            }
        }
    }

    // A game may consist of headers only (no movetext, no result token).
    // The visitor still expects end_headers before end_game.
    if in_headers {
        visitor.end_headers();
    }
    visitor.end_game();
}

fn is_token_delimiter(c: u8) -> bool {
    matches!(
        c,
        b' ' | b'\n'
            | b'\r'
            | b'\t'
            | b'['
            | b']'
            | b'{'
            | b'}'
            | b'('
            | b')'
            | b';'
            | b'*'
            | b'.'
            | b'!'
            | b'?'
            | b'$'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecordingVisitor {
        events: Vec<String>,
    }

    impl Visitor for RecordingVisitor {
        fn begin_headers(&mut self) {
            self.events.push("begin_headers".to_string());
        }
        fn header(&mut self, key: &[u8], value: &[u8]) {
            self.events.push(format!(
                "header {}={}",
                String::from_utf8_lossy(key),
                String::from_utf8_lossy(value)
            ));
        }
        fn end_headers(&mut self) {
            self.events.push("end_headers".to_string());
        }
        fn san_token(&mut self, token: &[u8]) {
            self.events
                .push(format!("san {}", String::from_utf8_lossy(token)));
        }
        fn comment(&mut self, comment: &[u8]) {
            self.events
                .push(format!("comment {}", String::from_utf8_lossy(comment)));
        }
        fn outcome(&mut self, outcome: Outcome) {
            self.events.push(format!("outcome {:?}", outcome));
        }
        fn end_game(&mut self) {
            self.events.push("end_game".to_string());
        }
    }

    fn events(pgn: &str) -> Vec<String> {
        let mut visitor = RecordingVisitor::default();
        parse_game(pgn.as_bytes(), &mut visitor);
        visitor.events
    }

    fn sans(pgn: &str) -> Vec<String> {
        events(pgn)
            .iter()
            .filter_map(|e| e.strip_prefix("san ").map(str::to_owned))
            .collect()
    }

    #[test]
    fn test_move_numbers_without_space() {
        assert_eq!(sans("1.e4 e5 2.Nf3 Nc6 1-0"), ["e4", "e5", "Nf3", "Nc6"]);
    }

    #[test]
    fn test_black_move_number_continuation() {
        assert_eq!(
            sans("1.e4 1...e5 2. Nf3 2... Nc6 *"),
            ["e4", "e5", "Nf3", "Nc6"]
        );
    }

    #[test]
    fn test_attached_annotations() {
        let pgn = "1. e4!? e5?! 2. Nf3! Nc6?? 3. Bb5$14 a6 *";
        assert_eq!(sans(pgn), ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
    }

    #[test]
    fn test_standalone_nags_and_annotations() {
        assert_eq!(
            sans("1. e4 $1 e5 $139 2. Nf3 !? Nc6 ?? *"),
            ["e4", "e5", "Nf3", "Nc6"]
        );
    }

    #[test]
    fn test_castling_with_zeros_delivered_raw() {
        // Zeros-castling normalization is semantic and lives in the visitor
        // (see visitor.rs); the tokenizer delivers the tokens verbatim.
        let pgn = "1. 0-0 0-0-0 2. 0-0+ 0-0-0# 1-0";
        assert_eq!(sans(pgn), ["0-0", "0-0-0", "0-0+", "0-0-0#"]);
    }

    #[test]
    fn test_check_suffix_kept() {
        assert_eq!(sans("1. e4+ e5# *"), ["e4+", "e5#"]);
    }

    #[test]
    fn test_bom_stripped() {
        let pgn = "\u{feff}[Event \"x\"]\n\n1. e4 *";
        let evs = events(pgn);
        assert!(evs.contains(&"header Event=x".to_string()));
        assert_eq!(sans(pgn), ["e4"]);
    }

    #[test]
    fn test_crlf() {
        let pgn = "[Event \"x\"]\r\n[Site \"y\"]\r\n\r\n1. e4 e5 1-0\r\n";
        let evs = events(pgn);
        assert!(evs.contains(&"header Event=x".to_string()));
        assert!(evs.contains(&"header Site=y".to_string()));
        assert_eq!(sans(pgn), ["e4", "e5"]);
        assert!(evs.contains(&"outcome WhiteWins".to_string()));
    }

    #[test]
    fn test_tag_value_escapes() {
        // Escaped quotes/backslashes delimit correctly; value passed raw (undecoded).
        let pgn = "[Event \"a \\\"quoted\\\" name\"]\n[Site \"back\\\\slash\"]\n\n*";
        let evs = events(pgn);
        assert!(evs.contains(&"header Event=a \\\"quoted\\\" name".to_string()));
        assert!(evs.contains(&"header Site=back\\\\slash".to_string()));
    }

    #[test]
    fn test_semicolon_line_comment() {
        assert_eq!(sans("1. e4 ; rest ignored e5 Nf3\nc5 *"), ["e4", "c5"]);
    }

    #[test]
    fn test_escape_line_at_input_start() {
        let pgn = "% ignored line\n[Event \"x\"]\n\n1. e4 *";
        let evs = events(pgn);
        assert!(evs.contains(&"header Event=x".to_string()));
        assert_eq!(sans(pgn), ["e4"]);
    }

    #[test]
    fn test_escape_line_mid_game() {
        assert_eq!(sans("1. e4\n% skipped e5\nc5 *"), ["e4", "c5"]);
    }

    #[test]
    fn test_percent_mid_line_is_not_escape() {
        // '%' not at line start is not an escape; the junk token is delivered
        // raw like any other token (the visitor decides how to handle it) and
        // tokenizing continues to the end of the game.
        let pgn = "1. e4 %junk e5 *";
        assert_eq!(sans(pgn), ["e4", "%junk", "e5"]);
    }

    #[test]
    fn test_variations_skipped() {
        let pgn = "1. e4 (1. d4 {inner} d5 (1... Nf6 2. c4) 2. c4) e5 2. Nf3 1-0";
        assert_eq!(sans(pgn), ["e4", "e5", "Nf3"]);
        let evs = events(pgn);
        assert!(!evs.iter().any(|e| e.starts_with("comment")));
        assert!(evs.contains(&"outcome WhiteWins".to_string()));
    }

    #[test]
    fn test_result_token_inside_variation_ignored() {
        let pgn = "1. e4 (1. d4 1-0) e5 (1... c5 *) 2. Nf3 *";
        assert_eq!(sans(pgn), ["e4", "e5", "Nf3"]);
        let evs = events(pgn);
        assert_eq!(evs.iter().filter(|e| e.starts_with("outcome")).count(), 1);
        assert!(evs.contains(&"outcome Unknown".to_string()));
    }

    #[test]
    fn test_trailing_game_ignored_after_result() {
        let pgn = "[Event \"a\"]\n\n1. e4 1-0\n\n[Event \"b\"]\n\n1. d4 0-1";
        let evs = events(pgn);
        assert!(evs.contains(&"header Event=a".to_string()));
        assert!(!evs.contains(&"header Event=b".to_string()));
        assert_eq!(sans(pgn), ["e4"]);
        assert!(evs.contains(&"outcome WhiteWins".to_string()));
        assert!(!evs.contains(&"outcome BlackWins".to_string()));
    }

    #[test]
    fn test_blank_line_terminates_movetext() {
        // Without a result token, a blank line still ends the game
        // (parity with pgn-reader).
        assert_eq!(sans("1. e4 e5\n\n2. Nf3 Nc6"), ["e4", "e5"]);
        assert_eq!(sans("1. e4 e5\r\n\r\n2. Nf3"), ["e4", "e5"]);
    }

    #[test]
    fn test_headers_only_game_flushes_end_headers() {
        let evs = events("[Event \"x\"]\n[Site \"y\"]\n");
        assert_eq!(
            evs,
            [
                "begin_headers",
                "header Event=x",
                "header Site=y",
                "end_headers",
                "end_game"
            ]
        );
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(events(""), ["begin_headers", "end_headers", "end_game"]);
    }

    #[test]
    fn test_unterminated_comment_is_lenient() {
        // Intentional divergence: pgn-reader raises a hard error here.
        let evs = events("1. e4 {never closed");
        assert!(evs.contains(&"comment never closed".to_string()));
        assert_eq!(sans("1. e4 {never closed"), ["e4"]);
    }

    #[test]
    fn test_garbage_token_delivered_raw() {
        let pgn = "1. e4 xyzzy9 e5 *";
        assert_eq!(sans(pgn), ["e4", "xyzzy9", "e5"]);
    }

    #[test]
    fn test_stray_closing_delimiters_no_hang() {
        assert_eq!(sans("1. e4 } ] e5 *"), ["e4", "e5"]);
    }

    #[test]
    fn test_comment_between_headers_and_moves() {
        let pgn = "[Event \"x\"]\n\n{pre-game comment} 1. e4 *";
        let evs = events(pgn);
        assert!(evs.contains(&"comment pre-game comment".to_string()));
        assert_eq!(sans(pgn), ["e4"]);
    }
}
