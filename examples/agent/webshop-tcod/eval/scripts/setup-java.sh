#!/usr/bin/env bash
# Source from a launcher after REPO_ROOT has been resolved.
for argument in "$@"; do
  if [[ "${argument}" == "--check-config" ]]; then
    return 0
  fi
done

webshop_java_default="${REPO_ROOT}/data/webshop-opd/java"
if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" && -f "${JAVA_HOME}/lib/server/libjvm.so" ]]; then
  export JAVA_HOME
elif [[ -x "${webshop_java_default}/bin/java" && -f "${webshop_java_default}/lib/server/libjvm.so" ]]; then
  if [[ -n "${JAVA_HOME:-}" ]]; then
    printf 'JAVA_HOME=%s is unavailable; using %s\n' "${JAVA_HOME}" "${webshop_java_default}" >&2
  fi
  export JAVA_HOME="${webshop_java_default}"
else
  printf 'Java 11 is required. Set JAVA_HOME to an installed JDK containing bin/java and lib/server/libjvm.so, or install it at %s.\n' "${webshop_java_default}" >&2
  return 1
fi

# Keep the executable and pyjnius library on the same selected Java installation.
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export PATH="${JAVA_HOME}/bin:${PATH}"
