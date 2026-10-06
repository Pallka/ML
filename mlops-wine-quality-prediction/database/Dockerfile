FROM postgres:15-alpine

COPY init.sql /docker-entrypoint-initdb.d/
RUN mkdir -p /data && chmod 755 /data

EXPOSE 5432
