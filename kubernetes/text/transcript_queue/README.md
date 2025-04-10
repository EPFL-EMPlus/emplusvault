Queue Producer
==============

This is a simple producer that reads from the database table entries_to_queue and puts them into a queue. 
After that the transcript consumers will read from the queue and process the entries.