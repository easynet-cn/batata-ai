use batata_ai_core::rag::Chunker;

/// Recursive, structure-aware chunker.
///
/// Splits on a priority list of separators (`\n\n\n` → `\n\n` → `\n`
/// → `. ` → ` ` → char), recursing into the next separator only when
/// a chunk still exceeds `max_window`. Merges adjacent small chunks up
/// to `max_window` so the embedder isn't wasted on tiny fragments.
///
/// This is the recommended chunker for Markdown, HTML (after loader
/// conversion to text), and source code. For pure prose of uniform
/// length, `FixedWindowChunker` is simpler and cheaper.
pub struct RecursiveChunker {
    max_window: usize,
    separators: Vec<&'static str>,
}

impl RecursiveChunker {
    pub fn new(max_window: usize) -> Self {
        assert!(max_window > 0, "max_window must be > 0");
        Self {
            max_window,
            // Ordered from "most structural" (paragraph breaks) to
            // "most granular" (characters).
            separators: vec!["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        }
    }

    fn split_recursive(&self, text: &str, sep_idx: usize) -> Vec<String> {
        if text.chars().count() <= self.max_window {
            return if text.is_empty() {
                Vec::new()
            } else {
                vec![text.to_string()]
            };
        }
        if sep_idx >= self.separators.len() {
            // Last resort: hard char split.
            return hard_char_split(text, self.max_window);
        }
        let sep = self.separators[sep_idx];
        let parts: Vec<&str> = if sep.is_empty() {
            // "" separator ⇒ character granularity fallback.
            return hard_char_split(text, self.max_window);
        } else {
            text.split(sep).collect()
        };
        let mut out = Vec::new();
        for part in parts {
            if part.chars().count() <= self.max_window {
                if !part.is_empty() {
                    out.push(part.to_string());
                }
            } else {
                out.extend(self.split_recursive(part, sep_idx + 1));
            }
        }
        self.merge_small(out)
    }

    fn merge_small(&self, pieces: Vec<String>) -> Vec<String> {
        let mut merged: Vec<String> = Vec::new();
        for piece in pieces {
            match merged.last_mut() {
                Some(last) if last.chars().count() + piece.chars().count() + 1 <= self.max_window => {
                    if !last.is_empty() {
                        last.push('\n');
                    }
                    last.push_str(&piece);
                }
                _ => merged.push(piece),
            }
        }
        merged
    }
}

fn hard_char_split(text: &str, window: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = (i + window).min(chars.len());
        out.push(chars[i..end].iter().collect::<String>());
        i = end;
    }
    out
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        self.split_recursive(text, 0)
    }
}

/// Fixed-size character window with overlap.
///
/// Character-based (not byte-based) so multi-byte UTF-8 content is safe.
/// Token-aware chunking can be added later as a separate impl.
pub struct FixedWindowChunker {
    window: usize,
    overlap: usize,
}

impl FixedWindowChunker {
    pub fn new(window: usize, overlap: usize) -> Self {
        assert!(window > overlap, "window must be larger than overlap");
        Self { window, overlap }
    }
}

impl Default for FixedWindowChunker {
    fn default() -> Self {
        Self::new(512, 64)
    }
}

impl Chunker for FixedWindowChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let step = self.window - self.overlap;
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + self.window).min(chars.len());
            out.push(chars[start..end].iter().collect::<String>());
            if end == chars.len() {
                break;
            }
            start += step;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        let c = FixedWindowChunker::new(10, 2);
        assert!(c.chunk("").is_empty());
    }

    #[test]
    fn single_chunk_when_text_shorter_than_window() {
        let c = FixedWindowChunker::new(10, 2);
        assert_eq!(c.chunk("abc"), vec!["abc".to_string()]);
    }

    #[test]
    fn overlapping_windows() {
        let c = FixedWindowChunker::new(5, 2);
        let out = c.chunk("abcdefghij");
        // window=5, step=3: positions 0..5, 3..8, 6..10
        assert_eq!(out, vec!["abcde", "defgh", "ghij"]);
    }

    #[test]
    fn recursive_respects_paragraph_breaks() {
        let text =
            "Section one content.\n\nSection two content.\n\nSection three content is here.";
        let c = RecursiveChunker::new(35);
        let out = c.chunk(text);
        // Each section < 35 chars, so each becomes its own chunk (after
        // merge; small chunks may combine).
        assert!(!out.is_empty());
        for piece in &out {
            assert!(piece.chars().count() <= 35, "chunk too big: {piece}");
        }
        // Content preserved.
        let joined = out.join(" ");
        assert!(joined.contains("Section one"));
        assert!(joined.contains("Section three"));
    }

    #[test]
    fn recursive_falls_back_to_chars_for_oversized() {
        let c = RecursiveChunker::new(8);
        // A single word exceeding the window should be hard-split.
        let out = c.chunk("abcdefghijklmnop");
        for piece in &out {
            assert!(piece.chars().count() <= 8);
        }
        assert!(out.join("").contains("abcdefghijklmnop"));
    }
}
