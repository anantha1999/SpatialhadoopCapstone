#!/usr/bin/expect
spawn ./sendJarToServer.sh
expect "user@10.10.1.149's password:"
send "password\r";
interact