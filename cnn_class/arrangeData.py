working_train_dir = "train/"
working_test_dir = "test/"
if (os.path.isdir(working_train_dir) == False):
    os.mkdir(working_train_dir)
    print("created " + working_train_dir)
else:
    print(working_train_dir + " exists")
if (os.path.isdir(working_test_dir) == False):
    os.mkdir(working_test_dir)
    print("created " + working_test_dir)
else:
    print(working_test_dir + " exists")

artist_dir = working_train_dir + 'picasso/'
not_artist_dir = working_train_dir + 'not-picasso/'
if (os.path.isdir(artist_dir) == False):
    os.mkdir(artist_dir)
    print("created " + artist_dir)
else:
    print(artist_dir + " exists")
if (os.path.isdir(not_artist_dir) == False):
    os.mkdir(not_artist_dir)
    print("created " + not_artist_dir)
else:
    print(not_artist_dir + " exists")

# same for test data 
test_artist_dir = working_test_dir + 'picasso/'
test_not_artist_dir = working_test_dir + 'not-picasso/'
if (os.path.isdir(test_artist_dir) == False):
    os.mkdir(test_artist_dir)
    print("created " + test_artist_dir)
else:
    print(test_artist_dir + " exists")
if (os.path.isdir(test_not_artist_dir) == False):
    os.mkdir(test_not_artist_dir)
    print("created " + test_not_artist_dir)
else:
    print(test_not_artist_dir + " exists")
