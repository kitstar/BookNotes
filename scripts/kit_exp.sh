#!/bin/bash

### constant parameter
work_path="~/run/dnn"



setup () 
{
    echo "Call Setup"
}


usage ()
{
    echo "Call Usage"
}



############################
# main ()                  #

case "${1}" in
"setup") setup
;;

"") usage
;;

*) echo "Invalid command!"
;;

esac
