#!/usr/bin/expect
spawn ssh user@10.10.1.149
expect "user@10.10.1.149's password:"
send "password\r";
interact