#!/usr/bin/env bash

# Render one YAML configuration with the repository root supplied by launchers.
render_config_args() {
  local default_config="$1"
  shift

  CONFIG_SOURCE="${default_config}"
  CONFIG_ARGS=()
  while (($#)); do
    case "$1" in
      --config)
        if (($# < 2)); then
          echo "--config requires a path" >&2
          return 2
        fi
        CONFIG_SOURCE="$2"
        shift 2
        ;;
      --config=*)
        CONFIG_SOURCE="${1#--config=}"
        shift
        ;;
      *)
        CONFIG_ARGS+=("$1")
        shift
        ;;
    esac
  done

  if ! command -v envsubst >/dev/null 2>&1; then
    echo "envsubst is required to render WebShop configuration templates." >&2
    echo "Install the gettext package and retry." >&2
    return 1
  fi
  if [[ ! -f "${CONFIG_SOURCE}" ]]; then
    echo "Configuration template not found: ${CONFIG_SOURCE}" >&2
    return 1
  fi
  : "${PROJECT_ROOT:?PROJECT_ROOT must be set by the launcher}"

  RENDERED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/webshop-tcod-config.XXXXXX")"
  export PROJECT_ROOT
  if ! envsubst '$PROJECT_ROOT' < "${CONFIG_SOURCE}" > "${RENDERED_CONFIG}"; then
    rm -f -- "${RENDERED_CONFIG}"
    RENDERED_CONFIG=""
    return 1
  fi
}
