"""Optional integrations for external agent SDKs.

The package-level namespace deliberately contains no client construction or
network side effects.  Concrete integrations live in a provider-specific
subpackage (currently ``openai``), where transport, session identity, tool
execution, and response formatting are kept behind explicit adapter objects.
Applications can therefore import the SDK package while deciding separately
which provider, credentials, event loop, and resource owner to use.
"""
