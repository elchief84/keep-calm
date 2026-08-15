/* WordPiece tokenizer for distilbert-base-multilingual-cased.
 *
 * Loads a vocab.json mapping token -> id (extracted from tokenizer.json),
 * then implements the WordPiece greedy longest-match algorithm.
 */

export class WordPieceTokenizer {
  constructor(vocab) {
    this.vocab = vocab; // token -> id
    this.clsId = vocab["[CLS]"];
    this.sepId = vocab["[SEP]"];
    this.padId = vocab["[PAD]"];
    this.unkId = vocab["[UNK]"] ?? vocab["<unk>"];
    this.maxCharsPerToken = 100;
  }

  // Pre-tokenize: split on whitespace and strip off punctuation, matching
  // the DistilBERT tokenizer's basic tokenization closely enough for chat text.
  basicTokenize(text) {
    const out = [];
    for (const raw of text.split(/\s+/)) {
      if (!raw) continue;
      // Split alphanumeric runs from surrounding punctuation.
      for (const token of raw.match(/[A-Za-z0-9À-ÿ]+|[^\sA-Za-z0-9À-ÿ]/g) ?? []) {
        out.push(token);
      }
    }
    return out;
  }

  // Greedy longest-match WordPiece on a single word.
  wordpiece(word) {
    if (word.length > this.maxCharsPerToken) {
      return [String(this.unkId)];
    }
    if (Object.prototype.hasOwnProperty.call(this.vocab, word)) {
      return [String(this.vocab[word])];
    }
    const tokens = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let found = null;
      while (start < end) {
        const sub = (start > 0 ? "##" : "") + word.slice(start, end);
        if (Object.prototype.hasOwnProperty.call(this.vocab, sub)) {
          found = sub;
          break;
        }
        end--;
      }
      if (found === null) {
        tokens.push(String(this.unkId));
        break;
      }
      tokens.push(String(this.vocab[found]));
      start += found.startsWith("##") ? found.length - 2 : found.length;
    }
    return tokens;
  }

  // Encode a sentence into input_ids + attention_mask (max_length = 256).
  encode(text, maxLength = 256) {
    const words = this.basicTokenize(text);
    const ids = [this.clsId];
    for (const word of words) {
      for (const idStr of this.wordpiece(word)) {
        ids.push(Number(idStr));
      }
    }
    ids.push(this.sepId);
    if (ids.length > maxLength) {
      ids.splice(maxLength - 1, ids.length - maxLength + 1, this.sepId);
    }
    const mask = new Array(maxLength).fill(0);
    for (let i = 0; i < ids.length && i < maxLength; i++) mask[i] = 1;
    while (ids.length < maxLength) ids.push(this.padId);
    return { input_ids: ids, attention_mask: mask };
  }
}

export async function loadTokenizer(url = "vocab.json") {
  const resp = await fetch(url);
  const vocab = await resp.json();
  return new WordPieceTokenizer(vocab);
}
