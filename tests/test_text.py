from berlin_labels.text import canon_nh, tokenize_cuisines


def test_canon_nh_umlauts_and_punct():
    assert canon_nh("Köllnische-Heide ") == "koellnischeheide"
    assert canon_nh("Mitte") == "mitte"
    assert canon_nh("Straße 123") == "strasse123"


def test_tokenize_cuisines_keeps_nationals_only():
    tokens = tokenize_cuisines("Italian; Pizza; Japanese; Döner; Vietnamese")
    assert set(tokens) == {"italian", "japanese", "vietnamese"}

