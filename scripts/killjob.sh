#!/bin/bash


OPT="$1"
PROCESS_PATTERN="$2"

USER=$(whoami)

echo "Username:" $USER
echo $'\n\n**********'

if [ "$OPT" = "l" ]; then

    pgrep -a -l -u "$USER" -f "$PROCESS_PATTERN"

# Do not kill if PROCESS_PATTERN is blank
elif [ "$OPT" = "KILL" ] && [ ! -z `sed -e "s/\ //g" <<< "$PROCESS_PATTERN"` ]; then

    pgrep -a -l -u "$USER" -f "$PROCESS_PATTERN"
    pkill -9 -u "$USER" -f "$PROCESS_PATTERN"

else

    echo "Invalid syntax."
    echo "Syntax: killjob <option> <process pattern>"
    echo "<option> = l: List processes"
    echo "<option> = KILL: Kill processes"
    echo "<process pattern> = process command contains this pattern (MUST if <option> is KILL)"

fi

echo $'\n'
