#!/usr/bin/env node
// The memchat-ts command. Kept apart from main.ts so tests can import main
// without starting a session.

import { main } from "./main.js";

process.exitCode = await main(process.argv.slice(2));
