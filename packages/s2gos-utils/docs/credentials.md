# Credentials

* Required to access files and datasets stored in authenticated remote locations.
* Secrets should never appear in version control or serialized files.
* S2GOS separates credential IDs (safe to store) from actual secrets (stored locally).

## Credential ID

* Identifier used to look up credentials at runtime.
* Use descriptive names indicating purpose or data source.
* Examples: `earthdatahub`, `s3ovh`, `my_institution`.

## Supported Authentication Types

### BasicAuth (HTTP)

| Field | Required | Description |
|-------|----------|-------------|
| `username` | Yes | HTTP username |
| `password` | Yes | HTTP password or token |


### S3

| Field | Required | Description |
|-------|----------|-------------|
| `key` | Yes | Access key ID |
| `secret` | Yes | Secret access key |
| `endpoint_url` | No | For S3-compatible services (not needed for AWS) |

## Credential Providers

* Responsible for retrieving credentials from a credential ID.

### Environment Variables

* Format: `S2GOS_CREDENTIALS__<credential_id>__<FIELD>`
* BasicAuth example:
    ```bash
    export S2GOS_CREDENTIALS__earthdatahub__TYPE=basic_auth
    export S2GOS_CREDENTIALS__earthdatahub__USERNAME=myuser
    export S2GOS_CREDENTIALS__earthdatahub__PASSWORD=mytoken
    ```
* S3 example:
    ```bash
    export S2GOS_CREDENTIALS__s3bucket__TYPE=s3
    export S2GOS_CREDENTIALS__s3bucket__KEY=AKIAIOSFODNN7EXAMPLE
    export S2GOS_CREDENTIALS__s3bucket__SECRET=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    export S2GOS_CREDENTIALS__s3bucket__ENDPOINT_URL=https://s3.eu-west-1.amazonaws.com
    ```

### Dynaconf (`.secrets.yaml`)

* Reads credentials from `.secrets.yaml` in the project root.
* **Never commit this file** - it should be gitignored.
* Environment variables take precedence over `.secrets.yaml`.
* See [dynaconf secrets documentation](https://www.dynaconf.com/secrets/).

```yaml
# .secrets.yaml
credentials:
  earthdatahub:
    type: basic_auth
    username: your_username
    password: your_token

  s3ovh:
    type: s3
    key: your_access_key
    secret: your_secret_key
    endpoint_url: https://s3.de.io.cloud.ovh.net  # Optional
```

## See Also

* [Data Access Layer](data_access_layer.md) - Using credentials with PathRef.