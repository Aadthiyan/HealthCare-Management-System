import smtplib

email = input("Sender Email: ")
receiver_email = input("Receiver Email: ")

sub = input("Subject: ")
message = input("Message: ")

text = f"Subject: {sub}\n\n{message}"

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
