import Tkinter as tk
import os
import sys
import tkFileDialog
from Tkinter import *
import ttk
import pandas as pd
import numpy as np
import got as got
from got import models

LARGE_FONT = ("Verdana", 12, "bold")
SMALL_FONT = ("Verdana", 9, "bold")


class MainPage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)  ##
        tk.Tk.wm_title(self, "Padawan 1.0 Beta")
        
        # Icon and geometry

        tk.Tk.wm_iconbitmap(self, default="padawan1.ico")
        self.geometry('{}x{}'.format(1000, 800))
        self.configure(background='black')
        
        # Dict to store tweets, some globals
        
        global tweets
        global n
        n = 0
        tweets = {}
        
        # Search criteria variables

        search_name = StringVar()
        since_name = StringVar()
        until_name = StringVar()
        load_tweets = IntVar()
        csv_name = StringVar()
        jump_tweets = IntVar()
            
        def pull_data():
            global tweetCriteria
            print("criteria fetched")
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
                search_name.get()).setSince(since_name.get()).setUntil(
                until_name.get()).setMaxTweets(load_tweets.get())
            t = got.manager.TweetManager.getTweets(tweetCriteria)
            print("Got data");
            print("Starting FOR LOOP")
            for i in xrange(0, len(t)):
                tweets[i] = {}
                tweets[i]["date"] = t[i].date
                tweets[i]["username"] = t[i].username
                tweets[i]["retweets"] = t[i].retweets
                tweets[i]["text"] = t[i].text
                tweets[i]["mentions"] = t[i].mentions
                tweets[i]["hashtags"] = t[i].hashtags
                tweets[i]["rating"] = 0
                tweets[i]["useful"] = 0
            df = pd.DataFrame.from_dict(tweets, orient='index')
            print("writer")  
            df.to_csv(csv_name.get() + ".csv",encoding='utf-8', index=False)
            
        def start_label():
            global label_df
            label_df = pd.read_csv(load_file_bt.cget('text'))
            global row
            row = 0
            display_tweet()
            
        def display_tweet():
            current_date_label.config(
                    text="Date: %s" % label_df.iloc[row]['date'])
            current_username_label.config(
                    text="Username: %s" % label_df.iloc[row]['username'])
            current_retweets_label.config(
                    text="Retweets: %s" % label_df.iloc[row]['retweets'])
            current_tweet_label.config(
                    text="Tweet: %s" % label_df.iloc[row]['text'])
            current_mentions_label.config(
                    text="Mentions: %s" % label_df.iloc[row]['mentions'])
            current_hashtags_label.config(
                    text="Hashtags: %s" % label_df.iloc[row]['hashtags'])
            current_tweet_number.config(
                    text="Current tweet number: %s" % str(row+1))

        def create_dataframe():
            label_df.to_csv(output_dir_but["text"] + "/" + output_name.get() + 
                            ".csv",encoding='utf-8', index=False)

        def positive(self):
            global row
            label_df.at[row, 'rating'] = 1
            label_df.at[row, 'useful'] = 1
            print(label_df.iloc[row])
            row += 1
            display_tweet()

        def negative(self):
            global row
            label_df.at[row, 'rating'] = -1
            label_df.at[row, 'useful'] = 1
            print(label_df.iloc[row])
            row += 1
            display_tweet()

        def useless(self):
            global row
            label_df.at[row, 'rating'] = 0
            label_df.at[row, 'useful'] = 0
            print(label_df.iloc[row])
            row += 1
            display_tweet()
            
        # Directory Functions, Buttons and Labels
        output_dir_text = 'Select output folder'

        def openDirectory():
            dir_name = tkFileDialog.askdirectory(parent=self, initialdir='/home/', title=dir_text)
            dir_but["text"] = str(dir_name) if dir_name else dir_text

        def outputDirectory():
            output_dir_name = tkFileDialog.askdirectory(parent=self, initialdir='/home/', title=output_dir_text)
            output_dir_but["text"] = str(output_dir_name) if output_dir_name else output_dir_text
            
        def load_file():
            file_name = tkFileDialog.askopenfilename(parent=self, initialdir='/home/', title=load_file_txt)
            load_file_bt.config(text = str(file_name) if file_name else load_file_txt)
            
        def set_row():
            global row
            row = jump_tweets.get()
            display_tweet()
            
        # Bind functions

        MainPage.bind(self, "<Left>", positive)
        MainPage.bind(self, "<Right>", negative)
        MainPage.bind(self, "<Key-Down>", useless)
        
        # Initialize criteria

        tweetCriteria = got.manager.TweetCriteria().setUsername('ethereum').setMaxTweets(1)
        try:
            tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
        except:
            IndexError

        # Frames

        directories = tk.Frame(self)
        directories.configure(background='black')
        directories.pack(side=RIGHT, anchor=NW, fill='both', expand=False)
        directories.grid_rowconfigure(0, weight=1)
        directories.grid_columnconfigure(0, weight=1)

        entries = tk.Frame(self)
        entries.configure(background='black')
        entries.pack(side=LEFT, anchor=NW)

        options = Frame(self, width=500, height=500)
        options.configure(background='black')
        options.pack(side=LEFT, anchor=NW, fill=X)
        
        # Search options

        search_label = Label(entries, text="Search terms", font=LARGE_FONT, fg="orange", bg="black")
        search_label.pack(pady=10, padx=10, anchor=NW)
        search_name_entry = Entry(entries, textvariable=search_name)
        search_name_entry.pack(side=TOP, padx=10, anchor=NW, fill='both')

        since_label = Label(entries, text="Search since", font=LARGE_FONT, fg="orange", bg="black")
        since_label.pack(pady=10, padx=10, anchor=NW)
        since_name_entry = Entry(entries, textvariable=since_name)
        since_name_entry.pack(side=TOP, padx=10, anchor=NW, fill='both')

        until_label = Label(entries, text="Search until", font=LARGE_FONT, fg="orange", bg="black")
        until_label.pack(pady=10, padx=10, anchor=NW)
        until_name_entry = Entry(entries, textvariable=until_name)
        until_name_entry.pack(side=TOP, padx=10, anchor=NW, fill='both')

        load_tweets_label = Label(entries, text="Load first n tweets:", font=LARGE_FONT, fg="orange", bg="black")
        load_tweets_label.pack(pady=10, padx=10, anchor=NW)
        load_tweets_entry = Entry(entries, textvariable=load_tweets)
        load_tweets_entry.pack(side=TOP, padx=10, anchor=NW, fill='both')
        
        csv_name_label = Label(entries, text="Save .csv file as:", font=LARGE_FONT, fg="orange", bg="black")
        csv_name_label.pack(pady=10, padx=10, anchor=NW)
        csv_name_entry = Entry(entries, textvariable=csv_name)
        csv_name_entry.pack(side=TOP, padx=10, anchor=NW, fill='both')

        start_but = ttk.Button(entries, text="Pull tweets to csv file", command=lambda: pull_data())
        start_but.pack(pady=10, padx=10, anchor=NW, fill='both')

        # Directory Buttons and Labels
        output_dir_text = 'Select output folder'

        dir_label = Label(directories, text="Save labelled .csv", font=LARGE_FONT, fg="orange", bg="black")
        dir_label.pack(pady=10, padx=10, anchor=NW)

        output_dir_label = Label(directories, text="Save in:", font=LARGE_FONT, fg="orange", bg="black")
        output_dir_label.pack(pady=10, padx=10, anchor=NW)
        output_dir_but = ttk.Button(directories, text=output_dir_text, command=outputDirectory)
        output_dir_but.pack(pady=10, padx=10, anchor=NW, fill='both')

        save_as_label = Label(directories, text="Save as:", font=LARGE_FONT, fg="orange", bg="black")
        save_as_label.pack(pady=10, padx=10, anchor=NW)

        output_name = StringVar()
        output_name_entry = Entry(directories, textvariable=output_name)
        output_name_entry.pack(side=TOP, padx=10, anchor=NW, fill='both')

        df_but = ttk.Button(directories, text="Output .csv file", command=lambda: create_dataframe())
        df_but.pack(pady=10, padx=10, anchor=NW, fill='both')
        
        load_file_label = Label(options, text="Load file for labelling:", font=LARGE_FONT, fg="orange", bg="black")
        load_file_label.pack(pady=0, padx=10, anchor=NW)
        load_file_txt = "Load File"

        load_file_bt = ttk.Button(options, text=load_file_txt, command=load_file)
        load_file_bt.pack(pady=10, padx=10, anchor=NW)
            
        start_labelling_bt = ttk.Button(options, text="Start Labelling", command=lambda: start_label())
        start_labelling_bt.pack(pady=10, padx=10, anchor=NW)
        
        # Displayed Tweet
        
        current_date_label = Label(options, text="Date: ", font=LARGE_FONT, fg="orange", bg="black")
        current_date_label.pack(pady=10, padx=10, anchor=NW)
        current_username_label = Label(options, text="Username: ", font=LARGE_FONT, fg="orange", bg="black")
        current_username_label.pack(pady=10, padx=10, anchor=NW)
        current_retweets_label = Label(options, text="Retweets: ", font=LARGE_FONT, fg="orange", bg="black")
        current_retweets_label.pack(pady=10, padx=10, anchor=NW)
        current_tweet_label = Label(options, text="Tweet: ", wraplength=400, font=LARGE_FONT, fg="orange", bg="black")
        current_tweet_label.pack(pady=10, padx=10, anchor=NW, expand=True)
        current_mentions_label = Label(options, text="Mentions: ", font=LARGE_FONT, fg="orange", bg="black")
        current_mentions_label.pack(pady=10, padx=10, anchor=NW)
        current_hashtags_label = Label(options, text="Hashtags: ", font=LARGE_FONT, fg="orange", bg="black")
        current_hashtags_label.pack(pady=10, padx=10, anchor=NW)
        current_tweet_number = Label(options, text="Current tweet number: %s" % n, font=SMALL_FONT, fg="orange",
                                     bg="black")
        current_tweet_number.pack(pady=10, padx=10, anchor=SW)
            
        jump_tweets_bt = ttk.Button(options, text="Jump", command = lambda: set_row())
        jump_tweets_bt.pack(side=BOTTOM, padx=10, pady=10, anchor=SW)
        jump_tweets_entry = Entry(options, textvariable=jump_tweets)
        jump_tweets_entry.pack(side=BOTTOM, padx=10, pady=10, anchor=SW, fill='both')        
        jump_tweets_label = Label(options, text="Jump to tweet no. :", font=LARGE_FONT, fg="orange", bg="black")
        jump_tweets_label.pack(side=BOTTOM, padx=10, pady=10, anchor=SW)

        # Keybinds

        binds = Frame(directories)
        binds.configure(background='black')
        binds.pack(side=BOTTOM)
        binds_label = Label(binds, text="LEFT: POSITIVE", font=LARGE_FONT, fg="orange", bg="black")
        binds_label.pack(padx=10, anchor=W, fill='both', expand=True)
        binds_label = Label(binds, text="RIGHT: NEGATIVE", font=LARGE_FONT, fg="orange", bg="black")
        binds_label.pack(padx=10, anchor=W, fill='both', expand=True)
        binds_label = Label(binds, text="DOWN: USELESS", font=LARGE_FONT, fg="orange", bg="black")
        binds_label.pack(padx=10, anchor=W, fill='both', expand=True)

# Start program
app = MainPage()
app.mainloop()


