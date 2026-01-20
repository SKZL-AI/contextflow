/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly DEV: boolean
  readonly PROD: boolean
  readonly MODE: string
  readonly VITE_API_URL?: string
  readonly VITE_API_WS_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
