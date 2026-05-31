#!/usr/bin/env bash
#
# Thread budget helpers for CPU Slurm wrappers.
#
# These helpers detect the number of physical CPU cores visible on the node,
# cap requested worker threads to that limit, and emit a warning when a run
# would otherwise oversubscribe the host. The goal is to keep the default
# pipeline automatic and sensible while still making oversubscription visible
# in the logs.

oft_detect_physical_cores() {
  if command -v lscpu >/dev/null 2>&1; then
    local count
    count="$(
      lscpu -p=CORE,SOCKET 2>/dev/null \
        | awk -F, '
            $0 !~ /^#/ && NF >= 2 {
              key = $1 "," $2
              if (!seen[key]++) n++
            }
            END { print n + 0 }
          '
    )"
    if [ -n "${count:-}" ] && [ "${count}" -gt 0 ] 2>/dev/null; then
      printf '%s\n' "${count}"
      return 0
    fi
  fi

  if command -v getconf >/dev/null 2>&1; then
    local online
    online="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if [ -n "${online:-}" ] && [ "${online}" -gt 0 ] 2>/dev/null; then
      printf '%s\n' "${online}"
      return 0
    fi
  fi

  return 1
}

oft_cap_thread_budget() {
  # Usage:
  #   oft_cap_thread_budget REQUESTED LABEL
  # Prints the capped thread count to stdout. Returns 0 on success.
  local requested="$1"
  local label="${2:-thread budget}"
  local physical_cores

  if [ -z "${requested:-}" ] || [ "${requested}" -lt 1 ] 2>/dev/null; then
    requested=1
  fi

  if physical_cores="$(oft_detect_physical_cores)"; then
    if [ "${requested}" -gt "${physical_cores}" ]; then
      echo "Warning: ${label} request of ${requested} threads exceeds ${physical_cores} physical cores visible on $(hostname); capping to ${physical_cores}." >&2
      requested="${physical_cores}"
    fi
    echo "${requested}"
    return 0
  fi

  echo "Warning: could not detect physical CPU cores on $(hostname); using requested ${requested} threads for ${label}." >&2
  echo "${requested}"
}
