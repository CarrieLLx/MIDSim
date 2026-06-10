"""
This module provides parsers for model responses, especially JSON responses.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

from loguru import logger
from pydantic import BaseModel

from ..core.model_response import ModelResponse

from .base import ParserBase

# Rednote-style note_id (24 hex); also used as fallback id scan in truncated JSON
_NOTE_ID_PATTERN = re.compile(r'"([0-9a-fA-F]{24})"')
_QUOTED_STRING_PATTERN = re.compile(r'"((?:[^"\\]|\\.)*)"')

DEFAULT_RECOMMENDATION_MATCH_FIELDS: Dict[str, str] = {
    "ranked_note_ids": "note_id_list",
    "selected_note_ids": "note_id_list",
}

DEFAULT_REACTION_MATCH_FIELDS: Dict[str, str] = {
    "memory_reflection": "string",
    "decision_reason": "string",
}

MEMORY_MATCH_FIELDS: Dict[str, str] = {
    "memory": "string",
    "Memory": "string",
}


class JsonBlockParser(ParserBase):
    """
    Parser for JSON objects in code blocks.
    
    This parser extracts JSON objects from markdown code blocks in
    model responses and parses them into Python objects.
    """
    
    def __init__(
        self,
        tag_start: str = "```json",
        tag_end: str = "```",
        content_hint: Optional[Any] = None
    ):
        """
        Initialize the parser.
        
        Args:
            tag_start: The start tag for the JSON block.
            tag_end: The end tag for the JSON block.
            content_hint: Optional hint for the expected content structure.
        """
        self.tag_start = tag_start
        self.tag_end = tag_end
        
        if content_hint is not None:
            if isinstance(content_hint, str):
                self.content_hint = content_hint
            else:
                self.content_hint = json.dumps(
                    content_hint,
                    ensure_ascii=False,
                    indent=2
                )
        else:
            self.content_hint = "```json\n{your_json_object}\n```"
            
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Extract and parse JSON from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with parsed JSON in the parsed field.
            
        Raises:
            ValueError: If JSON cannot be extracted or parsed.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # Find the JSON block
        start_idx = text.find(self.tag_start)
        
        if start_idx == -1:
            raise ValueError(f"Start tag '{self.tag_start}' not found in response")
        
        content_start = start_idx + len(self.tag_start)
        end_idx = text.find(self.tag_end, content_start)

        if end_idx == -1:
            json_content = text[content_start:].strip()
            logger.warning(
                "JSON block has no closing fence; parsing from opening tag to end of response "
                "(truncated or non-conformant model output)."
            )
        else:
            json_content = text[content_start:end_idx].strip()

        try:
            decoder = json.JSONDecoder()
            parsed_json, used = decoder.raw_decode(json_content)
            trailing = json_content[used:].strip()
            if trailing:
                logger.warning(
                    "JSON block contained extra data after first value; "
                    f"ignored {len(trailing)} chars: {trailing[:120]!r}"
                )
            response.parsed = parsed_json
            return response
        except json.JSONDecodeError as e:
            recovered = self._recover_parsed_json(json_content)
            if recovered is not None:
                n_ids = len(recovered.get("ranked_note_ids") or [])
                logger.warning(
                    f"JSON parse failed ({e}); recovered loose structure "
                    f"({n_ids} ranked_note_ids) from truncated output"
                )
                response.parsed = recovered
                return response
            raise ValueError(f"Failed to parse JSON: {e}") from e

    @staticmethod
    def _recover_parsed_json(json_content: str) -> Optional[Any]:
        if not json_content:
            return None
        marker = re.search(r'"ranked_note_ids"\s*:\s*\[', json_content)
        if not marker:
            return None
        chunk = json_content[marker.start():]
        ids: List[str] = []
        seen: set = set()
        for match in _NOTE_ID_PATTERN.finditer(chunk):
            nid = match.group(1)
            if nid in seen:
                continue
            seen.add(nid)
            ids.append(nid)
        if ids:
            return {"ranked_note_ids": ids}
        return None
    
    @property
    def format_instruction(self) -> str:
        """
        Get the format instruction for the model.
        
        Returns:
            A string with instructions on the expected response format.
        """
        return (
            f"Respond with a JSON object in a markdown code block as follows:\n"
            f"{self.tag_start}\n{self.content_hint}\n{self.tag_end}"
        )


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _slice_json_array_after_key(body: str, field_name: str) -> Optional[str]:
    """Locate \"field_name\": [ ... ]; if unclosed, return substring through end of body."""
    marker = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*\[',
        body,
        flags=re.IGNORECASE,
    )
    if not marker:
        return None
    start = marker.end() - 1
    depth = 0
    for i in range(start, len(body)):
        ch = body[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return body[start : i + 1]
    return body[start:]


def _extract_note_ids_from_text(text: str) -> List[str]:
    ids: List[str] = []
    seen: set = set()
    for match in _NOTE_ID_PATTERN.finditer(text):
        nid = match.group(1)
        if nid in seen:
            continue
        seen.add(nid)
        ids.append(nid)
    return ids


def _extract_quoted_strings_from_text(text: str) -> List[str]:
    return _dedupe_preserve_order(_QUOTED_STRING_PATTERN.findall(text))


def extract_json_string_field(text: str, field_name: str) -> Optional[str]:
    """Extract a string field from (possibly incomplete) JSON text."""
    if not text:
        return None
    marker = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*"',
        text,
        flags=re.IGNORECASE,
    )
    if not marker:
        return None
    i = marker.end()
    chars: List[str] = []
    while i < len(text):
        ch = text[i]
        if ch == '"':
            break
        if ch == "\\" and i + 1 < len(text):
            chars.append(ch)
            chars.append(text[i + 1])
            i += 2
            continue
        chars.append(ch)
        i += 1
    raw = "".join(chars)
    if not raw:
        return None
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return (
            raw.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def _extract_loose_json_body(text: str) -> str:
    tag_start, tag_end = "```json", "```"
    start_idx = text.find(tag_start)
    if start_idx != -1:
        content_start = start_idx + len(tag_start)
        end_idx = text.find(tag_end, content_start)
        if end_idx != -1:
            return text[content_start:end_idx].strip()
        return text[content_start:].strip()
    alt = text.find("```")
    if alt != -1:
        line_end = text.find("\n", alt)
        content_start = (line_end + 1) if line_end != -1 else alt + 3
        end_idx = text.find(tag_end, content_start)
        if end_idx != -1:
            return text[content_start:end_idx].strip()
        return text[content_start:].strip()
    return text


def recover_decisions_from_text(text: str) -> Optional[List[Any]]:
    """Try to salvage a decisions array from truncated or malformed JSON."""
    body = _extract_loose_json_body(text)
    marker = re.search(r'"decisions"\s*:\s*\[', body)
    if not marker:
        return None
    chunk = body[marker.end() - 1 :]
    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(chunk)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    objs: List[Any] = []
    for mobj in re.finditer(
        r'\{[^{}]*(?:"(?:note_id|blog_id|tweet_id)"|"comment")[^}]*\}',
        chunk,
    ):
        try:
            objs.append(json.loads(mobj.group(0)))
        except json.JSONDecodeError:
            continue
    return objs or None


def parse_json_block_loose(response: ModelResponse) -> Any:
    """
    Multi-strategy JSON parse: JsonBlockParser, then brace/bracket extraction.
    Raises ValueError if all strategies fail.
    """
    text = response.text
    if not text or not text.strip():
        raise ValueError("empty model response")
    parser = JsonBlockParser()
    try:
        return parser.parse(response).parsed
    except Exception:
        pass
    tag_start, tag_end = "```json", "```"
    start_idx = text.find(tag_start)
    if start_idx != -1:
        content_start = start_idx + len(tag_start)
        end_idx = text.find(tag_end, content_start)
        if end_idx != -1:
            chunk = text[content_start:end_idx].strip()
        else:
            chunk = text[content_start:].strip()
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(chunk)
            return parsed
        except json.JSONDecodeError:
            pass
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse JSON from model response")


def parse_llm_json_response(
    response: ModelResponse,
    *,
    field_match_fields: Optional[Dict[str, str]] = None,
    recover_decisions: bool = False,
    empty_fallback: Any = None,
) -> Tuple[Any, str, Optional[str]]:
    """
    Unified LLM JSON parse with FieldMatchParser fallback.

    Returns (parsed, method, error_note).
    method is one of: json_block, field_match, decisions_recover, empty.
    """
    error_note: Optional[str] = None
    partial: Optional[Dict[str, Any]] = None

    try:
        return parse_json_block_loose(response), "json_block", None
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        error_note = str(e)

    if field_match_fields:
        try:
            fm = FieldMatchParser(fields=field_match_fields)
            partial = fm.parse(response).parsed
            if partial:
                if recover_decisions and "decisions" not in partial:
                    decisions = recover_decisions_from_text(response.text or "")
                    if decisions:
                        merged = dict(partial)
                        merged["decisions"] = decisions
                        return merged, "decisions_recover", error_note
                return partial, "field_match", error_note
        except ValueError:
            pass

    if recover_decisions:
        decisions = recover_decisions_from_text(response.text or "")
        if decisions:
            base = dict(partial) if partial else {}
            base["decisions"] = decisions
            return base, "decisions_recover", error_note

    if partial:
        return partial, "field_match", error_note

    return empty_fallback, "empty", error_note


def parse_memory_response(
    response: ModelResponse,
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Parse memory field from LLM response with layered fallback.
    Returns (memory_text_or_none, method, error_note).
    """
    error_note: Optional[str] = None
    text = response.text or ""

    try:
        parsed = parse_json_block_loose(response)
        if isinstance(parsed, dict):
            mem = parsed.get("memory") or parsed.get("Memory")
            if mem is None and len(parsed) == 1:
                only_v = next(iter(parsed.values()))
                if isinstance(only_v, str) and only_v.strip():
                    mem = only_v.strip()
            if mem is not None and (not isinstance(mem, str) or mem.strip()):
                return str(mem), "json_block", None
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        error_note = str(e)

    try:
        fm = FieldMatchParser(fields=MEMORY_MATCH_FIELDS)
        parsed = fm.parse(response).parsed
        mem = parsed.get("memory") or parsed.get("Memory")
        if mem is not None and (not isinstance(mem, str) or mem.strip()):
            return str(mem), "field_match", error_note
    except ValueError:
        pass

    mem = extract_json_string_field(text, "memory") or extract_json_string_field(
        text, "Memory"
    )
    if mem and mem.strip():
        return mem.strip(), "field_extract", error_note

    return None, "none", error_note or "memory field not found"


class FieldMatchParser(ParserBase):
    """
    Match and extract named fields from model output without requiring valid JSON.
    Useful for truncated recommendation responses (ranked_note_ids, etc.).
    """

    SUPPORTED_TYPES = frozenset({"note_id_list", "string_list", "bool", "string"})

    def __init__(
        self,
        fields: Dict[str, str],
        tag_start: str = "```json",
        tag_end: str = "```",
    ):
        if not fields:
            raise ValueError("fields must not be empty")
        for field_type in fields.values():
            if field_type not in self.SUPPORTED_TYPES:
                raise ValueError(f"Unsupported field type: {field_type}")
        self.fields = fields
        self.tag_start = tag_start
        self.tag_end = tag_end

    def parse(self, response: ModelResponse) -> ModelResponse:
        text = response.text
        if not text:
            raise ValueError("Response text is empty")

        body = self._extract_response_body(text)
        parsed: Dict[str, Any] = {}
        for field_name, field_type in self.fields.items():
            value = self._extract_field(body, field_name, field_type)
            if value is not None and value != [] and value != "":
                parsed[field_name] = value

        if not parsed:
            primary_field, primary_type = next(iter(self.fields.items()))
            if primary_type == "note_id_list":
                fallback_ids = _extract_note_ids_from_text(body)
                if fallback_ids:
                    logger.warning(
                        f"FieldMatchParser: no key matched for {primary_field!r}; "
                        f"fallback scan found {len(fallback_ids)} note ids"
                    )
                    parsed[primary_field] = fallback_ids

        if not parsed:
            raise ValueError(
                f"No fields matched in response: {list(self.fields.keys())}"
            )

        response.parsed = parsed
        return response

    def _extract_response_body(self, text: str) -> str:
        return _extract_loose_json_body(text)

    def _extract_field(
        self, body: str, field_name: str, field_type: str
    ) -> Optional[Any]:
        if field_type == "note_id_list":
            chunk = _slice_json_array_after_key(body, field_name)
            if chunk:
                return _extract_note_ids_from_text(chunk)
            return None
        if field_type == "string_list":
            chunk = _slice_json_array_after_key(body, field_name)
            if chunk:
                return _extract_quoted_strings_from_text(chunk)
            return None
        if field_type == "string":
            return extract_json_string_field(body, field_name)
        marker = re.search(
            rf'"{re.escape(field_name)}"\s*:\s*(true|false)',
            body,
            flags=re.IGNORECASE,
        )
        if not marker:
            return None
        raw = marker.group(1)
        if field_type == "bool":
            return raw.lower() == "true"
        return None


class JsonDictParser(JsonBlockParser):
    """
    Parser for JSON dictionaries with field filtering capabilities.
    
    This parser extends JsonBlockParser to handle dictionaries specifically
    and provides methods to filter fields for different purposes.
    """
    
    def __init__(
        self,
        tag_start: str = "```json",
        tag_end: str = "```",
        content_hint: Optional[Any] = None,
        required_keys: Optional[List[str]] = None,
        keys_to_content: Union[str, bool, Sequence[str]] = True,
        keys_to_metadata: Union[str, bool, Sequence[str]] = False,
        schema: Optional[BaseModel] = None
    ):
        """
        Initialize the parser.
        
        Args:
            tag_start: The start tag for the JSON block.
            tag_end: The end tag for the JSON block.
            content_hint: Optional hint for the expected content structure.
            required_keys: List of keys that must be present in the parsed JSON.
            keys_to_content: Keys to include in content output.
            keys_to_metadata: Keys to include in metadata output.
            schema: Optional Pydantic model to validate the parsed JSON.
        """
        super().__init__(tag_start, tag_end, content_hint)
        
        self.required_keys = required_keys or []
        self.keys_to_content = keys_to_content
        self.keys_to_metadata = keys_to_metadata
        self.schema = schema
        
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Parse and validate a JSON dictionary.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with parsed JSON dictionary in the parsed field.
            
        Raises:
            ValueError: If validation fails.
        """
        # First parse the JSON using parent class
        response = super().parse(response)
        
        # Ensure it's a dictionary
        if not isinstance(response.parsed, dict):
            raise ValueError(
                f"Expected a JSON dictionary, got {type(response.parsed).__name__}"
            )
        
        # Check required keys
        missing_keys = [key for key in self.required_keys if key not in response.parsed]
        if missing_keys:
            keys_str = ", ".join(missing_keys)
            raise ValueError(f"Missing required keys in response: {keys_str}")
        
        # Validate with schema if provided
        if self.schema is not None:
            try:
                validated_data = self.schema(**response.parsed).dict()
                response.parsed = validated_data
            except Exception as e:
                raise ValueError(f"Schema validation failed: {e}")
                
        return response
        
    def to_content(self, parsed_response: Dict) -> Union[str, Dict, None]:
        """
        Filter fields for content output.
        
        Args:
            parsed_response: The parsed dictionary to filter.
            
        Returns:
            Filtered content based on keys_to_content configuration.
        """
        return self._filter_by_keys(parsed_response, self.keys_to_content)
    
    def to_metadata(self, parsed_response: Dict) -> Union[str, Dict, None]:
        """
        Filter fields for metadata output.
        
        Args:
            parsed_response: The parsed dictionary to filter.
            
        Returns:
            Filtered metadata based on keys_to_metadata configuration.
        """
        return self._filter_by_keys(parsed_response, self.keys_to_metadata)
    
    def _filter_by_keys(
        self, 
        data: Dict, 
        keys: Union[str, bool, Sequence[str]]
    ) -> Union[str, Dict, None]:
        """
        Filter a dictionary by keys.
        
        Args:
            data: The dictionary to filter.
            keys: Key specification (True for all, False for none,
                  string for single key, sequence for multiple keys).
                  
        Returns:
            Filtered data based on keys configuration.
        """
        if isinstance(keys, bool):
            if keys:
                return data
            else:
                return None
                
        if isinstance(keys, str):
            return data.get(keys)
            
        # If keys is a sequence
        return {k: v for k, v in data.items() if k in keys} 