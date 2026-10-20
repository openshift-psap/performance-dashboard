"""Tests for custom_dropdown profile details and display variant logic."""

import pytest

from custom_dropdown import (
    PROFILE_DETAILS,
    _generate_display_variants,
    get_profile_details,
)


class TestGetProfileDetails:
    def test_exact_key(self):
        details = get_profile_details("1000/1000")
        assert details["name"] == "Balanced Profile"

    def test_empty_string(self):
        assert get_profile_details("") == {}

    def test_unknown_profile(self):
        assert get_profile_details("9999/9999") == {}

    def test_k_notation_1k_1k(self):
        details = get_profile_details("1k/1k")
        assert details["name"] == "Balanced Profile"

    def test_parenthesized_k_notation(self):
        details = get_profile_details("Profile A: Balanced (1k/1k)")
        assert details["name"] == "Balanced Profile"

    def test_rounded_k_notation_2k_resolves_to_2048(self):
        """2k decodes to 2000 but should match PROFILE_DETAILS['2048/128'] via variant fallback."""
        details = get_profile_details("2k/128")
        assert details["name"] == "Short Prefill-Heavy"
        assert details["prompt_tokens"] == "2048"

    def test_rounded_k_notation_512_2k_resolves_to_512_2048(self):
        """512/2k decodes to 512/2000 but should match '512/2048' via variant fallback."""
        details = get_profile_details("512/2k")
        assert details["name"] == "Heterogeneous"

    def test_multi_turn_alias(self):
        details = get_profile_details("Multi-turn")
        assert details["name"] == "Multi-turn Profile"
        assert details["turns"] == "5"

    def test_100k_1k(self):
        details = get_profile_details("100k/1k")
        assert details["name"] == "Long Context Prefill-Heavy"

    def test_8k_1k(self):
        details = get_profile_details("8k/1k")
        assert details["name"] == "Prefill-Heavy"


class TestGenerateDisplayVariants:
    def test_1000_1000_variants(self):
        variants = _generate_display_variants("1000/1000")
        assert "1k/1k" in variants
        assert "(1k/1k)" in variants

    def test_2048_128_includes_decoded_form(self):
        variants = _generate_display_variants("2048/128")
        assert "2000/128" in variants

    def test_512_2048_includes_decoded_form(self):
        variants = _generate_display_variants("512/2048")
        assert "512/2000" in variants

    def test_128_128_includes_multi_turn_alias(self):
        variants = _generate_display_variants("128/128")
        assert "Multi-turn" in variants

    def test_invalid_input(self):
        assert _generate_display_variants("invalid") == []


class TestProfileDetailsCompleteness:
    @pytest.mark.parametrize("key", PROFILE_DETAILS.keys())
    def test_required_fields(self, key):
        details = PROFILE_DETAILS[key]
        assert "name" in details
        assert "prompt_tokens" in details
        assert "output_tokens" in details
