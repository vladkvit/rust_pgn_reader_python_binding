use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{char, digit1, multispace1},
    combinator::{map, map_res, not, opt, peek, recognize},
    multi::{many0, many1},
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};

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
    Text(&'a str),
    Tag(ParsedTag),
}

pub fn parse_comments(input: &str) -> IResult<&str, Vec<CommentContent>> {
    many0(alt((
        // Attempt to parse a known structured tag first
        map(parse_structured_tag, CommentContent::Tag),
        // If not a known tag, but looks like a tag (e.g. [%unknown ...]), parse as text
        map(
            recognize(delimited(tag("[%"), is_not("]"), char(']'))),
            |s: &str| CommentContent::Text(s),
        ),
        // Otherwise, parse as regular text content. This must not be empty.
        map(
            recognize(many1(alt((
                is_not("["), // Takes any char except '['
                // Takes a '[' if it's NOT followed by '%' (to allow "[abc]" as text)
                recognize(preceded(char('['), peek(not(char('%'))))),
            )))),
            |s: &str| CommentContent::Text(s),
        ),
    )))
    .parse(input)
}

/// Parser for a complete tag like [%eval ...] or [%clk ...]
fn parse_structured_tag(input: &str) -> IResult<&str, ParsedTag> {
    delimited(
        tag("[%"),
        alt((parse_eval_content, parse_clk_content)),
        char(']'),
    )
    .parse(input)
}

/// Parses the content of an eval tag, e.g., "eval 12.3" or "eval #3"
fn parse_eval_content(input: &str) -> IResult<&str, ParsedTag> {
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
fn parse_clk_content(input: &str) -> IResult<&str, ParsedTag> {
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
fn parse_signed_float(input: &str) -> IResult<&str, f64> {
    map_res(
        recognize(pair(
            opt(alt((char('+'), char('-')))),
            recognize(pair(digit1, opt(preceded(char('.'), digit1)))),
        )),
        |s: &str| s.parse::<f64>(),
    )
    .parse(input)
}

/// Parser for a mate value, e.g., "#-3", "#5"
fn parse_mate_value(input: &str) -> IResult<&str, i32> {
    preceded(
        char('#'),
        map_res(
            recognize(pair(opt(char('-')), digit1)), // Recognizes signed integer
            |s: &str| s.parse::<i32>(),
        ),
    )
    .parse(input)
}

/// Parser for a time value in H:M:S format, e.g., "12:34:56" or "1:2:3.45"
fn parse_hms_time(input: &str) -> IResult<&str, (u32, u8, f64)> {
    map(
        (
            map_res(digit1, |s: &str| s.parse::<u32>()), // Hours
            char(':'),
            map_res(digit1, |s: &str| s.parse::<u8>()), // Minutes (0-255, typically 0-59)
            char(':'),
            parse_seconds_with_fraction, // Seconds with optional fraction
        ),
        |(h, _, m, _, s)| (h, m, s),
    )
    .parse(input)
}

/// Parser for seconds, which can be an integer or have a fractional part
fn parse_seconds_with_fraction(input: &str) -> IResult<&str, f64> {
    map_res(
        recognize(pair(digit1, opt(preceded(char('.'), digit1)))),
        |s: &str| s.parse::<f64>(),
    )
    .parse(input)
}

/// Parser for one or more whitespace characters (spaces, newlines, tabs, etc.)
fn spacing(input: &str) -> IResult<&str, &str> {
    multispace1(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comments1() {
        let input = "[%eval 123] some text [%clk 12:34:56]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::Tag(ParsedTag::Eval(123.0)),
                CommentContent::Text(" some text "),
                CommentContent::Tag(ParsedTag::ClkTime {
                    hours: 12,
                    minutes: 34,
                    seconds: 56.0
                })
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_comments2() {
        let input = "[%clk 12:34:56] some text ";
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
                CommentContent::Text(" some text ")
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_comment_fractional_sec() {
        let input = "[%clk 12:34:56.0123]";
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
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_structured_tag_eval() {
        let input = "[%eval 123]";
        let result = parse_structured_tag(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Eval(123.0));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_structured_tag_eval_mate() {
        let input = "[%eval #-3]";
        let result = parse_structured_tag(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Mate(-3));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_structured_tag_clk() {
        let input = "[%clk 1:2:3.45]";
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
        assert_eq!(remaining, "");
    }

    // clk expects a particular format: H:M:S - this test ensures incorrect formats fail
    #[test]
    fn test_tag_parser_incorrect_clk_value() {
        let input = "[%clk 123]"; // Incorrect format for clk time
        let result = parse_structured_tag(input);
        assert!(
            result.is_err(),
            "Parser should fail for incorrect clk format"
        );
    }

    #[test]
    fn test_tag_parser_incorrect_eval_value() {
        let input = "[%eval notanumber]";
        let result = parse_structured_tag(input);
        assert!(
            result.is_err(),
            "Parser should fail for non-numeric eval value"
        );
    }

    #[test]
    fn test_parse_clk_content_correct() {
        let input = "clk 12:34:56";
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
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_clk_content_incorrect_name() {
        // This tests if parse_clk_content correctly fails if "clk" is not the prefix
        let input = "eval 123";
        let result = parse_clk_content(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_eval_content_correct_float() {
        let input = "eval -42.5";
        let result = parse_eval_content(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Eval(-42.5));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_eval_content_correct_mate() {
        let input = "eval #5";
        let result = parse_eval_content(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, ParsedTag::Mate(5));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_signed_float_positive() {
        let input = "123.45";
        let result = parse_signed_float(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 123.45);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_signed_float_negative() {
        let input = "-0.5";
        let result = parse_signed_float(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, -0.5);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_mate_value_positive() {
        let input = "#7";
        let result = parse_mate_value(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 7);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_mate_value_negative() {
        let input = "#-2";
        let result = parse_mate_value(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, -2);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_hms_time_simple() {
        let input = "12:34:56";
        let result = parse_hms_time(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, (12, 34, 56.0));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_hms_time_fractional() {
        let input = "01:02:03.123";
        let result = parse_hms_time(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, (1, 2, 3.123));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_hms_time_invalid_char_in_hour() {
        let input = "1a:00:00";
        let result = parse_hms_time(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_hms_time_missing_minutes() {
        let input = "1::00";
        let result = parse_hms_time(input); // digit1 for minutes will fail on ':'
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_seconds_with_fraction_integer() {
        let input = "42";
        let result = parse_seconds_with_fraction(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 42.0);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_seconds_with_fraction_decimal() {
        let input = "3.141";
        let result = parse_seconds_with_fraction(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, 3.141);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_unknown_tag_is_parsed_as_text() {
        let input = "[%timestamp 12345] some text";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                // The whole unknown tag becomes a text element
                CommentContent::Text("[%timestamp 12345]"),
                CommentContent::Text(" some text") // Note: leading space is part of the text
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_clk_tag_with_newline_spacing() {
        let input = "[%clk\n0:09:36.2]";
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
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_eval_tag_with_newline_spacing() {
        let input = "[%eval\n-0.5]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, vec![CommentContent::Tag(ParsedTag::Eval(-0.5))]);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_text_containing_brackets_not_tag() {
        let input = "Text with [normal brackets] and then [%clk 1:2:3]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::Text("Text with [normal brackets] and then "),
                CommentContent::Tag(ParsedTag::ClkTime {
                    hours: 1,
                    minutes: 2,
                    seconds: 3.0
                })
            ]
        );
        assert_eq!(remaining, "");
    }
}
