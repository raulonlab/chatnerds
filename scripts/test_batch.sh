#!/bin/bash
input="./questions.txt"
while IFS= read -r line
do
#   command="chatnerds chat '$line'"
#   echo $command
    chatnerds chat "$line"
    sleep 1
done < "$input"
