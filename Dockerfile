FROM lucaplawliet/whisper-cli:latest

WORKDIR /app/whisperbot

RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN make

CMD ["./whisperbot"]