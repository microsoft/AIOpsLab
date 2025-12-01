# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for LiteLLM client exception handling."""

import unittest
from unittest.mock import patch, MagicMock


class TestLiteLLMExceptionHandling(unittest.TestCase):
    """Test safe exception handling in LiteLLM client."""

    def test_safe_get_exception_details_with_request_attribute(self):
        """Test that exception details are extracted safely when request attribute exists."""
        from clients.utils.llm import LiteLLMClient
        
        # Create a mock exception with request attribute
        mock_exception = Exception("API Error")
        mock_exception.request = {"url": "https://api.example.com"}
        mock_exception.response = {"status": 500}
        mock_exception.status_code = 500
        
        client = LiteLLMClient.__new__(LiteLLMClient)
        client.cache = None
        client.model = "test-model"
        
        # We need to import litellm to set it on the client
        import litellm
        client.litellm = litellm
        
        details = client._safe_get_exception_details(mock_exception)
        
        self.assertEqual(details["message"], "API Error")
        self.assertEqual(details["type"], "Exception")
        self.assertEqual(details["request"], {"url": "https://api.example.com"})
        self.assertEqual(details["response"], {"status": 500})
        self.assertEqual(details["status_code"], 500)

    def test_safe_get_exception_details_without_request_attribute(self):
        """Test that exception details are extracted safely when request attribute is missing.
        
        This tests the fix for the AttributeError: 'Exception' object has no attribute 'request'
        issue that occurs when LiteLLM encounters API errors.
        """
        from clients.utils.llm import LiteLLMClient
        
        # Create a plain exception without any custom attributes
        plain_exception = Exception("Simple Error")
        
        client = LiteLLMClient.__new__(LiteLLMClient)
        client.cache = None
        client.model = "test-model"
        
        import litellm
        client.litellm = litellm
        
        details = client._safe_get_exception_details(plain_exception)
        
        # Should safely return None for missing attributes instead of raising AttributeError
        self.assertEqual(details["message"], "Simple Error")
        self.assertEqual(details["type"], "Exception")
        self.assertIsNone(details["request"])  # Key fix: should be None, not raise AttributeError
        self.assertIsNone(details["response"])
        self.assertIsNone(details["status_code"])
        self.assertIsNone(details["llm_provider"])
        self.assertIsNone(details["model"])

    def test_safe_get_exception_details_with_partial_attributes(self):
        """Test exception details extraction with only some attributes present."""
        from clients.utils.llm import LiteLLMClient
        
        # Create an exception with only some custom attributes
        partial_exception = Exception("Partial Error")
        partial_exception.status_code = 400
        partial_exception.llm_provider = "openrouter"
        # Note: request and response are intentionally NOT set
        
        client = LiteLLMClient.__new__(LiteLLMClient)
        client.cache = None
        client.model = "test-model"
        
        import litellm
        client.litellm = litellm
        
        details = client._safe_get_exception_details(partial_exception)
        
        self.assertEqual(details["message"], "Partial Error")
        self.assertEqual(details["status_code"], 400)
        self.assertEqual(details["llm_provider"], "openrouter")
        self.assertIsNone(details["request"])  # Should be None, not error
        self.assertIsNone(details["response"])  # Should be None, not error


class TestGetAttrPatternForExceptions(unittest.TestCase):
    """Test the getattr pattern for safe exception attribute access.
    
    This validates that using getattr(exception, 'request', None) works correctly
    for the fix described in the issue.
    """

    def test_getattr_pattern_with_request_attribute(self):
        """Test getattr returns the attribute value when it exists."""
        class MockApiException(Exception):
            def __init__(self, message, request_data):
                super().__init__(message)
                self.request = request_data
        
        exc = MockApiException("API Error", {"url": "https://example.com"})
        
        # Using getattr pattern from the suggested fix
        request = getattr(exc, 'request', None)
        
        self.assertEqual(request, {"url": "https://example.com"})

    def test_getattr_pattern_without_request_attribute(self):
        """Test getattr returns None when the attribute doesn't exist.
        
        This is the key scenario that was causing the AttributeError in the original bug.
        """
        # Plain exception without request attribute
        exc = Exception("Simple Error")
        
        # Using getattr pattern from the suggested fix - should NOT raise AttributeError
        request = getattr(exc, 'request', None)
        
        self.assertIsNone(request)

    def test_getattr_pattern_prevents_attribute_error(self):
        """Test that getattr prevents AttributeError that was occurring."""
        exc = Exception("Error without request attr")
        
        # This would have raised AttributeError before the fix
        # AttributeError: 'Exception' object has no attribute 'request'
        try:
            request = getattr(exc, 'request', None)
            self.assertIsNone(request)
        except AttributeError:
            self.fail("getattr should not raise AttributeError")


if __name__ == "__main__":
    unittest.main()
