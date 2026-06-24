#!/bin/bash

echo "nameserver 8.8.8.8" > /etc/resolv.conf
ifconfig eth0 192.168.0.254
apt-get update -y
apt-get install -y isc-dhcp-server
echo "Aguardando"
sleep 3
echo "Finalizado"
/etc/init.d/isc-dhcp-server restart
