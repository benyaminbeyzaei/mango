// Record a short voice question with MediaRecorder: audio/webm (Chrome, Firefox) or audio/mp4 (Safari).
import { useEffect, useRef, useState } from "react"

const MAX_SECONDS = 60

export function useRecorder(onDone: (audio: Blob) => void) {
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState("")
  const recorder = useRef<MediaRecorder | null>(null)
  const cancelled = useRef(false)
  const done = useRef(onDone)
  const mounted = useRef(true)
  useEffect(() => {
    done.current = onDone
  })

  const stopRecorder = () => {
    if (recorder.current?.state === "recording") recorder.current.stop()
  }

  // leaving the page discards a recording in progress and releases the microphone
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      cancelled.current = true
      stopRecorder()
    }
  }, [])

  useEffect(() => {
    if (!recording) return
    const started = Date.now()
    const timer = setInterval(() => {
      const s = Math.floor((Date.now() - started) / 1000)
      setSeconds(s)
      if (s >= MAX_SECONDS) stopRecorder()
    }, 250)
    return () => clearInterval(timer)
  }, [recording])

  async function start() {
    setError("")
    if (
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      setError(
        "This browser cannot record audio here (recording needs a recent browser and https or localhost)."
      )
      return
    }
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (err) {
      const name = (err as DOMException).name
      setError(
        name === "NotAllowedError"
          ? "Microphone permission was denied. Allow it in the browser's site settings to ask by voice."
          : name === "NotFoundError"
            ? "No microphone was found."
            : `Could not start the microphone (${name}).`
      )
      return
    }
    if (!mounted.current) {
      stream.getTracks().forEach((t) => t.stop())
      return
    }
    const mimeType = ["audio/webm", "audio/mp4"].find((t) =>
      MediaRecorder.isTypeSupported(t)
    )
    const r = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    const chunks: Blob[] = []
    r.ondataavailable = (e) => chunks.push(e.data)
    r.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      setRecording(false)
      if (!cancelled.current)
        done.current(new Blob(chunks, { type: r.mimeType }))
    }
    cancelled.current = false
    recorder.current = r
    setSeconds(0)
    r.start()
    setRecording(true)
  }

  const stop = stopRecorder
  const cancel = () => {
    cancelled.current = true
    stopRecorder()
  }

  return { recording, seconds, error, start, stop, cancel }
}
