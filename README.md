# datastreamer
server side: `./server`

`source .venv/bin/activate`


in `client/env` 

- `VITE_WS_URL=ws://localhost:9002` for dev
- `VITE_WS_URL=wss://thesaurus-process-jazz.ngrok-free.dev/` for prod

`ngrok http 9002`