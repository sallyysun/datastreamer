#!/bin/bash

set -e

echo "🚀 Building frontend..."
cd client
npm install
npm run build
cd ..

echo "🚀 Building server..."
make clean && make

echo "🚀 Starting server..."
./server