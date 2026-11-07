// Interactive-mode settings, kept in localStorage (the admin token lives in api/client.ts).
const DEFAULT_PRESET_KEY = "defaultPreset"
const CONFIRM_TRANSCRIPT_KEY = "confirmTranscript"

export function getDefaultPreset() {
  return localStorage.getItem(DEFAULT_PRESET_KEY) ?? "baseline"
}

export function setDefaultPreset(name: string) {
  localStorage.setItem(DEFAULT_PRESET_KEY, name)
}

export function getConfirmTranscript() {
  return localStorage.getItem(CONFIRM_TRANSCRIPT_KEY) === "true"
}

export function setConfirmTranscript(on: boolean) {
  localStorage.setItem(CONFIRM_TRANSCRIPT_KEY, String(on))
}
