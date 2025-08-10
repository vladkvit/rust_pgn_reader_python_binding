use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{char, digit1},
    combinator::{map, map_res, not, opt, peek, recognize},
    multi::{many0, many1},
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};
use std::borrow::Cow;
// use nom::character::complete::multispace1;
use nom::bytes::complete::take_while1;

#[derive(Debug, PartialEq)]
pub enum ParsedTag {
    Eval(f64),
    Mate(i32),
    ClkTime {
        hours: u32,
        minutes: u8,
        seconds: f64,
    },
}

#[derive(Debug, PartialEq)]
pub enum CommentContent<'a> {
    Text(Cow<'a, str>),
    Tag(ParsedTag),
}

fn as_str(bytes: &[u8]) -> Result<&str, std::str::Utf8Error> {
    std::str::from_utf8(bytes)
}

fn to_cow_str(bytes: &[u8]) -> CommentContent<'_> {
    CommentContent::Text(String::from_utf8_lossy(bytes))
}

pub fn parse_comments(input: &[u8]) -> IResult<&[u8], Vec<CommentContent<'_>>> {
    many0(alt((
        // Attempt to parse a known structured tag first
        map(parse_structured_tag, CommentContent::Tag),
        // If not a known tag, but looks like a tag (e.g. [%unknown ...]), parse as text
        map(
            recognize(delimited(tag("[%"), is_not("]"), char(']'))),
            to_cow_str,
        ),
        // Otherwise, parse as regular text content. This must not be empty.
        map(
            recognize(many1(alt((
                is_not([b'[']), // Takes any char except '['
                // Takes a '[' if it's NOT followed by '%' (to allow "[abc]" as text)
                recognize(preceded(char('['), peek(not(char('%'))))),
            )))),
            to_cow_str,
        ),
    )))
    .parse(input)
}

/// Parser for a complete tag like [%eval ...] or [%clk ...]
fn parse_structured_tag(input: &[u8]) -> IResult<&[u8], ParsedTag> {
    delimited(
        tag("[%"),
        alt((parse_eval_content, parse_clk_content)),
        char(']'),
    )
    .parse(input)
}

/// Parses the content of an eval tag, e.g., "eval 12.3" or "eval #3"
fn parse_eval_content(input: &[u8]) -> IResult<&[u8], ParsedTag> {
    preceded(
        tag("eval"),
        preceded(
            spacing,
            alt((
                map(parse_mate_value, ParsedTag::Mate),
                map(parse_signed_float, ParsedTag::Eval),
            )),
        ),
    )
    .parse(input)
}

/// Parses the content of a clk tag, e.g., "clk 1:23:45.6"
fn parse_clk_content(input: &[u8]) -> IResult<&[u8], ParsedTag> {
    preceded(
        tag("clk"),
        preceded(
            spacing,
            map(parse_hms_time, |(h, m, s)| ParsedTag::ClkTime {
                hours: h,
                minutes: m,
                seconds: s,
            }),
        ),
    )
    .parse(input)
}

/// Parser for a signed floating-point number, e.g., "-123.45", "+3.0", "7"
fn parse_signed_float(input: &[u8]) -> IResult<&[u8], f64> {
    map_res(
        map_res(
            recognize(pair(
                opt(alt((char('+'), char('-')))),
                recognize(pair(digit1, opt(preceded(char('.'), digit1)))),
            )),
            as_str,
        ),
        |s: &str| s.parse::<f64>(),
    )
    .parse(input)
}

/// Parser for a mate value, e.g., "#-3", "#5"
fn parse_mate_value(input: &[u8]) -> IResult<&[u8], i32> {
    preceded(
        char('#'),
        map_res(
            map_res(
                recognize(pair(opt(char('-')), digit1)), // Recognizes signed integer
                as_str,
            ),
            |s: &str| s.parse::<i32>(),
        ),
    )
    .parse(input)
}

/// Parser for a time value in H:M:S format, e.g., "12:34:56" or "1:2:3.45"
fn parse_hms_time(input: &[u8]) -> IResult<&[u8], (u32, u8, f64)> {
    map(
        (
            map_res(map_res(digit1, as_str), |s: &str| s.parse::<u32>()), // Hours
            char(':'),
            map_res(map_res(digit1, as_str), |s: &str| s.parse::<u8>()), // Minutes
            char(':'),
            parse_seconds_with_fraction, // Seconds with optional fraction
        ),
        |(h, _, m, _, s)| (h, m, s),
    )
    .parse(input)
}

/// Parser for seconds, which can be an integer or have a fractional part
fn parse_seconds_with_fraction(input: &[u8]) -> IResult<&[u8], f64> {
    map_res(
        map_res(
            recognize(pair(digit1, opt(preceded(char('.'), digit1)))),
            as_str,
        ),
        |s: &str| s.parse::<f64>(),
    )
    .parse(input)
}

/// Parser for one or more whitespace characters (spaces, newlines, tabs, etc.)
fn spacing(input: &[u8]) -> IResult<&[u8], &[u8]> {
    take_while1(|c: u8| c.is_ascii_whitespace())(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comments1() {
        let input = b"[%eval 123] some text [%clk 12:34:56]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::Tag(ParsedTag::Eval(123.0)),
                CommentContent::Text(Cow::from(" some text ")),
                CommentContent::Tag(ParsedTag::ClkTime {
                    hours: 12,
                    minutes: 34,
                    seconds: 56.0
                })
            ]
        );
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_comments2() {
        let input = b"[%clk 12:34:56] some text ";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::Tag(ParsedTag::ClkTime {
                    hours: 12,
                    minutes: 34,
                    seconds: 56.0
                }),
                CommentContent::Text(Cow::from(" some text "))
            ]
        );
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_comment_fractional_sec() {
        let input = b"[%clk 12:34:56.0123]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![CommentContent::Tag(ParsedTag::ClkTime {
                hours: 12,
                minutes: 34,
                seconds: 56.0123
            })]
        );
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_structured_tag_eval() {
        let input = b"[%eval 123]";
        let result = parse_structured_tag(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Eval(123.0));
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_structured_tag_eval_mate() {
        let input = b"[%eval #-3]";
        let result = parse_structured_tag(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Mate(-3));
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_structured_tag_clk() {
        let input = b"[%clk 1:2:3.45]";
        let result = parse_structured_tag(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            ParsedTag::ClkTime {
                hours: 1,
                minutes: 2,
                seconds: 3.45
            }
        );
        assert_eq!(remaining, b"");
    }

    // clk expects a particular format: H:M:S - this test ensures incorrect formats fail
    #[test]
    fn test_tag_parser_incorrect_clk_value() {
        let input = b"[%clk 123]"; // Incorrect format for clk time
        let result = parse_structured_tag(input);
        assert!(
            result.is_err(),
            "Parser should fail for incorrect clk format"
        );
    }

    #[test]
    fn test_tag_parser_incorrect_eval_value() {
        let input = b"[%eval notanumber]";
        let result = parse_structured_tag(input);
        assert!(
            result.is_err(),
            "Parser should fail for non-numeric eval value"
        );
    }

    #[test]
    fn test_parse_clk_content_correct() {
        let input = b"clk 12:34:56";
        let result = parse_clk_content(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            ParsedTag::ClkTime {
                hours: 12,
                minutes: 34,
                seconds: 56.0
            }
        );
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_clk_content_incorrect_name() {
        // This tests if parse_clk_content correctly fails if "clk" is not the prefix
        let input = b"eval 123";
        let result = parse_clk_content(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_eval_content_correct_float() {
        let input = b"eval -42.5";
        let result = parse_eval_content(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Eval(-42.5));
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_eval_content_correct_mate() {
        let input = b"eval #5";
        let result = parse_eval_content(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Mate(5));
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_signed_float_positive() {
        let input = b"123.45";
        let result = parse_signed_float(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 123.45);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_signed_float_negative() {
        let input = b"-0.5";
        let result = parse_signed_float(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, -0.5);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_mate_value_positive() {
        let input = b"#7";
        let result = parse_mate_value(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 7);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_mate_value_negative() {
        let input = b"#-2";
        let result = parse_mate_value(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, -2);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_hms_time_simple() {
        let input = b"12:34:56";
        let result = parse_hms_time(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, (12, 34, 56.0));
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_hms_time_fractional() {
        let input = b"01:02:03.123";
        let result = parse_hms_time(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, (1, 2, 3.123));
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_hms_time_invalid_char_in_hour() {
        let input = b"1a:00:00";
        let result = parse_hms_time(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_hms_time_missing_minutes() {
        let input = b"1::00";
        let result = parse_hms_time(input); // digit1 for minutes will fail on ':'
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_seconds_with_fraction_integer() {
        let input = b"42";
        let result = parse_seconds_with_fraction(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 42.0);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_parse_seconds_with_fraction_decimal() {
        let input = b"3.141";
        let result = parse_seconds_with_fraction(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 3.141);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_unknown_tag_is_parsed_as_text() {
        let input = b"[%timestamp 12345] some text";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                // The whole unknown tag becomes a text element
                CommentContent::Text(Cow::from("[%timestamp 12345]")),
                CommentContent::Text(Cow::from(" some text")) // Note: leading space is part of the text
            ]
        );
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_clk_tag_with_newline_spacing() {
        let input = b"[%clk\n0:09:36.2]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![CommentContent::Tag(ParsedTag::ClkTime {
                hours: 0,
                minutes: 9,
                seconds: 36.2
            })]
        );
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_eval_tag_with_newline_spacing() {
        let input = b"[%eval\n-0.5]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, vec![CommentContent::Tag(ParsedTag::Eval(-0.5))]);
        assert_eq!(remaining, b"");
    }

    #[test]
    fn test_text_containing_brackets_not_tag() {
        let input = b"Text with [normal brackets] and then [%clk 1:2:3]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::Text(Cow::from("Text with [normal brackets] and then ")),
                CommentContent::Tag(ParsedTag::ClkTime {
                    hours: 1,
                    minutes: 2,
                    seconds: 3.0
                })
            ]
        );
        assert_eq!(remaining, b"");
    }
}
