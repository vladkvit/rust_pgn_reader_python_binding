use memchr::memchr;
use shakmaty::san::{ParseSanError, SanPlus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    WhiteWins,
    BlackWins,
    Draw,
    Unknown,
}

pub trait Visitor {
    fn begin_game(&mut self);
    fn begin_headers(&mut self);
    fn header(&mut self, key: &[u8], value: &[u8]);
    fn end_headers(&mut self);
    fn san(&mut self, san_plus: SanPlus);
    fn san_error(&mut self, error: ParseSanError, token: &[u8]);
    fn comment(&mut self, comment: &[u8]);
    fn outcome(&mut self, outcome: Outcome);
    fn end_game(&mut self);
}

pub fn parse_game<V: Visitor>(mut bytes: &[u8], visitor: &mut V) {
    visitor.begin_game();

    // Strip UTF-8 BOM
    if bytes.starts_with(b"\xEF\xBB\xBF") {
        bytes = &bytes[3..];
    }

    let mut in_headers = true;
    visitor.begin_headers();

    let mut cursor = 0;
    let len = bytes.len();
    let mut variation_depth = 0;

    while cursor < len {
        let c = bytes[cursor];

        match c {
            b' ' | b'\n' | b'\r' | b'\t' => {
                cursor += 1;
            }
            b'[' => {
                if !in_headers {
                    // Start of a new game but we are already in movetext -> ignore trailing games
                    break;
                }
                cursor += 1;
                
                // Parse key
                let key_start = cursor;
                while cursor < len && bytes[cursor] != b' ' && bytes[cursor] != b'"' && bytes[cursor] != b']' {
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
            b';' | b'%' => {
                // Line comment or escape line
                if in_headers && c == b'%' {
                    // Escape lines can appear in headers too
                } else if in_headers {
                    in_headers = false;
                    visitor.end_headers();
                }
                if let Some(end_offset) = memchr(b'\n', &bytes[cursor..]) {
                    cursor += end_offset + 1;
                } else {
                    cursor = len;
                }
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
                if in_headers {
                    visitor.end_headers();
                }
                visitor.outcome(Outcome::Unknown);
                break; // End of game
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
                    cursor += 1;
                    continue;
                }

                // Check for move numbers or outcomes
                if token == b"1-0" {
                    visitor.outcome(Outcome::WhiteWins);
                    break;
                } else if token == b"0-1" {
                    visitor.outcome(Outcome::BlackWins);
                    break;
                } else if token == b"1/2-1/2" {
                    visitor.outcome(Outcome::Draw);
                    break;
                } else if memchr(b'.', token).is_some() || token.iter().all(|b| b.is_ascii_digit()) {
                    // Move number, skip
                } else if token == b"!" || token == b"?" || token == b"!!" || token == b"??" || token == b"!?" || token == b"?!" {
                    // Standalone suffix, ignore
                } else {
                    // It's a SAN token
                    if variation_depth == 0 {
                        match SanPlus::from_ascii(token) {
                            Ok(san_plus) => visitor.san(san_plus),
                            Err(e) => visitor.san_error(e, token),
                        }
                    }
                }
            }
        }
    }

    visitor.end_game();
}

fn is_token_delimiter(c: u8) -> bool {
    matches!(c, b' ' | b'\n' | b'\r' | b'\t' | b'[' | b']' | b'{' | b'}' | b'(' | b')' | b';' | b'*')
}
