rm /run/*_grpc.pid
monit
monit start all
tail -f /var/log/monit.log