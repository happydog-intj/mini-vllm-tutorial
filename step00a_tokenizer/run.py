from tokenizer import SimpleBPETokenizer

def main():
    tok = SimpleBPETokenizer()

    # 基础 encode/decode
    text = "Hello, world!"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"原文: {text!r}")
    print(f"Token IDs: {ids}")
    print(f"词表大小: {tok.vocab_size}")
    print(f"解码还原: {decoded!r}")
    assert decoded == text, f"解码不一致: {decoded!r} != {text!r}"

    # 字节级覆盖
    text2 = "你好世界"
    ids2 = tok.encode(text2)
    assert tok.decode(ids2) == text2, "中文 round-trip 失败"
    print(f"\n中文 round-trip OK: {text2!r} → {ids2} → {tok.decode(ids2)!r}")

    print("\n✅ step00a_tokenizer 通过")

if __name__ == "__main__":
    main()
