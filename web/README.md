# Serve Lab web

Vite + React + TypeScript frontend for Serve Lab. Tailwind CSS v4, shadcn/ui and components from
[The Gridcn](https://thegridcn.com) registry (copied into `src/components/thegridcn` and adapted to a
light "Tron claro" theme). Fonts are bundled, so the built app works fully offline.

## Build

```bash
cd web
npm install
npm run build        # type-check + build to web/dist
```

The Python server (`app/api.py`) serves `web/dist` at `/` and the JSON API under `/api` on
http://127.0.0.1:7860. Rebuild after any change here.

## Develop

```bash
npm run dev                 # http://localhost:5173, proxies /api to http://127.0.0.1:7860
VITE_MOCK=1 npm run dev     # no backend: fixtures from src/fixtures and a fake ~4 s job
```

In mock mode `?auto=jc-s15.mp4,jc-s4.MOV` runs those demos in order (handy for screenshots).

## Layout

- `src/api.ts`: typed API client (and the mock switch)
- `src/App.tsx`: state for the run and the session (React state only)
- `src/components/InputPanel.tsx`, `ResultPanel.tsx`, `MomentTiles.tsx`, `ProgressPanel.tsx`
- `src/index.css`: theme tokens, fonts, motion rules
