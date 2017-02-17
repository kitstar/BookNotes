#!/bin/bash

### Constantis
username="kit"
password="19870817"

host_ip="192.168.1.120"
machine_list=machine_list.txt
script_path="~/run/dnn"
python_path="~/anaconda2"



### UDFs
install_python()
{
    echo "TODO"
}



### Functions
setup () 
{
    install_python
}


copy_script()
{
    while IFS= read -r line; do
        if [ "${line}" != "${host_ip}" ]
        then
            echo "copying scripts to ${line}"
            ssh ${username}@{line} "mkdir ${script_path}"
            scp -r -C "${script_path}/*" "${username}@${line}:${script_path}"
        fi
    done < ${machine_list}
}


usage ()
{
    echo "Call Usage"
}



############################
# main ()                  #

case "${1}" in
"setup" | "st") setup
;;

"copyscript" | "cs") copy_script
;;


"") usage
;;

*) echo "Invalid command!"
;;

esac
