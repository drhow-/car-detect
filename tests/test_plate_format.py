from src.review.plate import _validate_plate_format


def test_arabic_gov_digits_valid():
    valid, flag = _validate_plate_format("دمشق 345678")
    assert valid is True
    assert flag is None


def test_arabic_gov_no_space_valid():
    valid, flag = _validate_plate_format("حلب123456")
    assert valid is True


def test_latin_gov_digits_valid():
    valid, flag = _validate_plate_format("DAMASCUS 345678")
    assert valid is True


def test_digits_only_valid():
    valid, flag = _validate_plate_format("345678")
    assert valid is True


def test_arabic_indic_digits_valid():
    valid, flag = _validate_plate_format("٣٤٥٦٧٨")
    assert valid is True


def test_empty_string_invalid():
    valid, flag = _validate_plate_format("")
    assert valid is False
    assert flag == "unreadable"


def test_random_text_invalid():
    valid, flag = _validate_plate_format("Hello World 123 ABC")
    assert valid is False
    assert flag == "hallucination_suspected"


def test_mixed_garbage_invalid():
    valid, flag = _validate_plate_format("XY-123-AB-456")
    assert valid is False
    assert flag == "hallucination_suspected"
