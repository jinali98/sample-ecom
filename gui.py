from email.header import Header
from tkinter import *
from chat import get_response, bot_name

MAIN_COLOR ="#9D5353"
DIVIDER_COLOR = "#DACC96"
SECOND_MAIN_COLOR="#EFEAD8"
SECONDARY_COLOR="#DACC96"
TEXT_COLOR="#632626"
HEADER_BG="#DACC96"
HEADER_TEXT_COLOR="#5F7161"
BUTTON_COLOR="#DACC96"


FONT_MAIN = 'Arial 12'
FONT_BOLD = 'Arial 14 bold'

class ChatBot:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
    
    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("COVID-19 SUPPORT - ChatBot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=550, height=600, bg=SECOND_MAIN_COLOR)

        # creating the header lable
        header_label = Label(self.window, text='Covid-19 Support', font=FONT_BOLD, pady=13, bg=BUTTON_COLOR, fg=TEXT_COLOR)
        header_label.place(relwidth=1)

        # creating a divider
        divider = Label(self.window, width=460, bg=DIVIDER_COLOR)
        divider.place(relwidth=1, rely=0.07, relheight=0.012)

        # creating a text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=SECOND_MAIN_COLOR, fg=TEXT_COLOR, font=FONT_MAIN, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # creating a scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window,  bg=MAIN_COLOR, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # creating a text entry widget
        self.msg_entry= Entry(bottom_label, bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=FONT_MAIN)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, relx=0.011, rely=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter)

        # button to send the msg
        send_button = Button(bottom_label, text="Send", bg=BUTTON_COLOR, fg=TEXT_COLOR, font=FONT_BOLD, width=15, command=lambda: self._on_enter(None))
        send_button.place(relwidth=0.22, relheight=0.06, relx=0.77, rely=0.008)



    def _on_enter(self, event):
        msg = self.msg_entry.get()
        self._insert_msg(msg, "Me")

    def _insert_msg(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        userText = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, userText)
        self.text_widget.configure(state=DISABLED)

        chatbotReply = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, chatbotReply)
        self.text_widget.configure(state=DISABLED)

# to scroll to the end all the time to see the latest msg
        self.text_widget.see(END)



if __name__ == "__main__":
    app = ChatBot()
    app.run()