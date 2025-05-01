#!/bin/bash
set -e

host="$1"
shift
cmd="$@"

until PGPASSWORD=root psql -h "$host" -U "postgres" -d "movies_db" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing command"
exec $cmd

