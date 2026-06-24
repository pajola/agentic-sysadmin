## Errors resolutions
### Error:
``` Err http://security.ubuntu.com oneiric-security Release.gpg
  Temporary failure resolving ‘security.ubuntu.com’ 
```
 Solution:  
 do a 'ping 8.8.8.8' at the terminals of every machine that downloads packages and then, after a successful ping, run the apt-get commands one by one at the terminal.  
 this is a common error at the inicialization of the DNS, DHCP and FTP machines.
 
 ### Errors related to name resolving  
 Solution:  
 just run the local dns configuration at the terminal of the machine going trough problems again, after the kathara inicialization, and things should be solved.

 ### Error (web):  
``` 
apache2: apr_sockaddr_info_get() failed for web
```  
Solution: 
inserir  
```
ping app01.admweb.empresa.com.br 
```
e após, tentar startar o apache novamente.
