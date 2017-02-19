#!/bin/bash

### Constantis
username="kit"
password="19870817"

host_ip="192.168.1.120"
machine_list=machine_list.txt
script_path="/home/kit/exp/dnn"
run_path="/home/kit/run/dnn"
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

auth()
{
    if [ ! -f "~/.ssh/id_rsa.pub" ]; then
        ssh-keygen -t rsa
    fi

    while IFS= read -r line; do
#        if [ "${line}" != "${host_ip}" ] && [ "${line:0:1}" != "#" ]
        if [ "${line:0:1}" != "#" ]
        then
            echo "copying authority to ${line}"
            ssh-copy-id -i ~/.ssh/id_rsa.pub "${line}"
        fi
    done < ${machine_list}
}


mkdirs()
{
    ssh -n ${1}@${2} "mkdir -p ${3}"
}


copy_script()
{
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            echo "copying scripts to ${line}"
            #ssh ${username}@${line} "mkdir -p ${run_path}"
            mkdirs $username $line $run_path
            scp -r -C ${script_path}/* ${username}@${line}:${run_path}
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
"auth") auth
;;

"setup" | "st") setup
;;

"copyscript" | "cs") copy_script
;;


"") usage
;;

*) echo "Invalid command!"
;;

esac
