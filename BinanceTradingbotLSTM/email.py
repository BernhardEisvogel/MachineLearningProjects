#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:16:20 2023

@author: be
"""
#python -m smtpd -n

from smtplib import SMTP
import datetime

debuglevel = 0

smtp = SMTP()
smtp.set_debuglevel(debuglevel)
smtp.connect('imap://imap.1und1.de', 26)
smtp.login('bernhard@eisvogel', 'fgz6-78/rE4p.')

from_addr = "John Doe <bernhard@eisvogel.net>"
to_addr = "bernhard.eisvogel@googlemail.com"

subj = "hello"
date = datetime.datetime.now().strftime( "%d/%m/%Y %H:%M" )

message_text = "Hello\nThis is a mail from your server\n\nBye\n"

msg = "From: %s\nTo: %s\nSubject: %s\nDate: %s\n\n%s" % ( from_addr, to_addr, subj, date, message_text )

smtp.sendmail(from_addr, to_addr, msg)
smtp.quit()