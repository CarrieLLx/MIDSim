"""Safe diagnostic details for model transport failures."""


def exception_details(exc):
    """Record exception types/causes and request metadata, never request headers."""
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        chain.append({'type': type(current).__name__, 'message': str(current)[:500]})
        current = current.__cause__ or current.__context__
    return {'chain': chain, 'status_code': getattr(exc, 'status_code', None),
            'request_id': getattr(exc, 'request_id', None)}
