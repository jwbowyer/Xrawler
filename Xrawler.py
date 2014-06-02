#Code mostly inspired by/borrowed from "Natural Language Processing with Python" (Bird, Klein, Loper)
#Need to install the imports in setup script
#%load_ext autoreload		%autoreload 2

import feedparser
import sys
import os
import nltk
import numpy as np
import fnmatch
import pickle
import time

from lxml.html import parse
from urllib2 import urlopen

#Can I train it to tell crackpottery from noncrackpottery (or distinguish vs snarxiv)?	davidsd.org/2010/03/the-snarxiv/
#Could ML using data from snarxiv.org/vs-arxiv/
#Use weights to classify shades of grey between "good" and "bad" classifications
#Consider the neutral class and topic models.
#include automated "similar words" to the keywords.
#I could assume that the probability of two keywords being x close together follows a Gaussian probability; then I can assign a probability (confidence limit) of a positive hit [actually, this is bad -- I need actual frequencies]

class papers(object):
	#www.pythoncentral.io/how-to-sort-a-list-tuple-or-object-with-sorted-in-python/

	def __init__(self,paper,weight):
		self.paper = paper
		self.weight = weight

	def __repr__(self):
		return '{}: {} {}'.format(self.__class__.__name__,self.paper,self.weight)


def getKey(paper):
	#return paper.weight
	return paper[1]	

def getSnarxiv():

   #Since snarxiv generates randomly each time one visits it, we do mc on this routine

   url = "http://snarxiv.org"
   time.sleep(np.random.randint(1,10))
   try:
    llog = feedparser.parse(url)
    return llog
   except:
    print "Could not grab from snarXiv! Please check your mirror.\n"
    return

def getManySnarxiv(mc=100):

  newdict = []
  for i in range(mc):
    newdict.append(llog['entries'][i] for i in range(0,nentries) )

  return newdict

def TrainSnarxiv(mc=100):

   #To do! Classification training based on many iterations of snarxiv.org/vs-arxiv
   #Per reload/click-through, we get 1 arxiv and 1 snarxiv paper; we also get the number of correct guesses we've made.
   #This value updates linearly, so it is trivial to find out whether we had a hit or a miss.
   #I'll need to be able to stream from javascript apps
   url="http://snarxiv.org/vs-arxiv/"
   print "Not yet implemented"

   return 0


def getTrainingData(keyword):

	classics=[]
	intros=[]
	other=[]		#use a random collection of non-<keyword> papers for this
	keyword=keyword.lower()

	if (keyword=="cmb") or (keyword=="cosmic microwaved background"):
		classics=["astro-ph/9603033"]
		#See the wiki: en.wikipedia.org/wiki/Cosmic_microwave_background_radiation
	if keyword=="b-modes":
		intros=["astro-ph/9904102","astro-ph/9706147","astro-ph/9810506"]
		classics=["astro-ph/9801285","astro-ph/0106174","astro-ph/0106536","astro-ph/9609170","astro-ph/9609169"]
	if (keyword=="quantum gravity"):
		intros=["1402.3586","1108.3269","0711.2445","gr-qc/0508120","gr-qc/0410054",
			"gr-qc/0108040","gr-qc/0004005","hep-th/9404019"]

	if (keyword=="quantum gravity"):
		[other,a]=getTrainingData("b-modes")
		del a
	#specials:1206.1192

	return [classics+intros,other]	#need a better set for "other"

def naiveSelect(TextList,weights,wtThresh=1e-10):

	Select = []
	for i in range(0,len(TextList)):
		if weights[i]>wtThresh:
			Select.append((TextList[i],weights[i]))
	Select = sorted(Select,reverse=True)
	return Select

def naiveWeighted(keywords,keyweights,titleShorts,abstractShorts,idShorts,wtThresh=1e-10,subset=0):

  nentries=len(titleShorts)
  finalList=[]

  #Count each keyword per title/abstract: counts[i][j]= [id][word]
  titleCounts = [ [ titleShorts[i].count(word) for word in keywords] for i in range(0,nentries)]
  titleWtMeans = [ np.average(titleCounts[i],weights=keyweights) for i in range(0,nentries)]
  #So the weighted mean counts give our first, most naive way of selecting preprints to read.

  #The wtMeans is the part that I can make more sophisticated. Two approaches are:
   #Use conditional probabilites
   #Compute the false-positive rate etc.

  #At the second level, I scan the abstracts for the keywords.
  abstractCounts = [ [ abstractShorts[i].count(word) for word in keywords] for i in range(0,nentries)]
  abstractWtMeans = [ np.average(abstractCounts[i],weights=keyweights) for i in range(0,nentries)]
	
  #Strip out those with weights below some threshold. Best to get combined list, rather than individual lists
  Select = []
  if subset==0:
   Select = naiveSelect(idShorts,titleWtMeans,wtThresh)  #Sort better: wiki.python.org/moin/HowTo/Sorting
  elif subset==1:
   Select = naiveSelect(idShorts,abstractWtMeans,wtThresh)
  elif subset==2:
   #Combine datasets
   combCounts = [ titleCounts[i]+abstractCounts[i] for i in range(0,nentries)]
   keyhalf = [ weight/2.0 for weight in keyweights ]
   keyweights = keyweights+keyhalf	#abstracts have lower weights
   combWtMeans = [ np.average(combCounts[i],weights=keyweights) for i in range(0,nentries)]
   Select = naiveSelect(idShorts,combWtMeans,wtThresh)

  return Select

def get_words_in_text(text):
	all_words=[]
	for (words, sentiment) in text:
		all_words.extend(words)
	return all_words

def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features=wordlist.keys()
	return word_features
	#return the ordered list of most frequent unique words

def extract_features(document):
	document_words=set(document)	#unordered unique collection
	features = {}
	for word in word_features:		#?
		features['contains(%s)' % word] = (word in document_words)
	return features

def to_index(rets):
	index = (1 + rets).cumprod()
	first_loc = max(index.notnull().argmax() - 1, 0)
	index.values[first_loc] = 1
	return index

def cleanTagData(papers,kword,force_get=0):

    abstr = []
    tset = []

    for trainer in papers:
     url="http://arxiv.org/abs/"+trainer

     fromFile=0
     ffile=data_dir+choice+"_trainingset_"+str(trainer.replace("/",""))+".pkl"
     if os.path.isfile(ffile)==1:
      if force_get==1:
       os.system("rm "+ffile)
      else:
       fromFile=1

     if fromFile==0:
      time.sleep(np.random.randint(1,10))
      try:
	parsed = parse(urlopen(url))
      except:
	print "Could not grab trainer from arXiv!"
	break

      doc = parsed.getroot()
      tables = doc.findall('.//blockquote')
      abstr.append(" ".join((tables[0].text_content()).split()[1:]))	#need to clean out tex code

      with open(ffile,"wb") as ff:
	pickle.dump(abstr,ff,pickle.HIGHEST_PROTOCOL)

     else:

      with open(ffile,"r") as ff:
	abstr = pickle.load(ff)

    nTraining=len(abstr)
    texRemove=["\\emph{","}"]			#List of tex codes to clean out.
    abstrFilt = [ " ".join([ word.lower() for word in abstr[i].split() if len(word)>2 ]) for i in range(0,nTraining) ]
    abstrFShorts= [ (abstrFilt[i].replace(texRemove[0],"")).replace(texRemove[1],"") for i in range(0,nTraining) ]

    #POS tags: CC=coordinating conjunction; RB=adverbs; IN=preposition; NN=noun; JJ=adjective.
    tokens = [ nltk.word_tokenize(abstrFShorts[i]) for i in range(0,nTraining) ]
    tagged = [ nltk.pos_tag(tokens[i]) for i in range(0,nTraining) ]

    #Filter out the following tags. Sometimes it gets these wrong! i.e., assigns 'kerr-black-hole-mirror'=JJ
    #filter_tags = ["IN","DT","RB","JJ","CC","PRP$","TO"]	#Could try combinations
    filter_tags = ["NN"]
    filtered = [ [word for (word,tag) in tagged[i] if (tag in filter_tags)] for i in range(0,nTraining) ]
    tset=[ (filtered[i],kword) for i in range(0,nTraining) ]
    del abstrFilt, abstrFShorts, tokens, tagged, filtered

    return tset


def getTrainingSet(keyword,force_get=0):

    [tPapers,oPapers] = getTrainingData(keyword)

    tset = cleanTagData(tPapers,keyword,force_get)
    tset += cleanTagData(oPapers,"other",force_get)
    
    return tset

def FindArXivRSS( keywords=["dark energy","unification","quantum gravity","CMB"], keyweights=[2,3,1,1],method=1,force_get=0):

 #Need to adjust for similar words, i.e., "gravity" also uses "gravitational"
 wtThresh=1e-6
 finalList = []
 tclassified = []
 aclassified = []
 global word_features

 #Create data folder
 global data_dir
 global choice
 data_dir="Xrawler_data/"
 d = os.path.dirname(data_dir)
 if not os.path.exists(d):
  os.makedirs(d)

 #Ideally, I want to grab the dictionary data from the web (checking for updates), and then io to/from a text or database file.
 sourceList=["astro-ph","cond-mat","gr-qc","hep-ex","hep-lat","hep-ph","hep-th","math-ph","nlin","nucl-ex","nucl-th","physics","quant-ph","math","corr","q-bio","q-fin","stats"]
 sourceListFullNames=["Astrophysics","Condensed Matter","General Relativity and Quantum Cosmology","High Energy Physics - Experiment","High Energy Physics - Lattice","High Energy Physics - Phenomenology","High Energy Physics - Theory","Mathematical Physics","Nonlinear Sciences","Nuclear Experiment","Nuclear Theory","Physics","Quantum Physics","Mathematics","Computing Research Repository","Quantitive Biology","Quantitive Finance","Statistics"]

 sourceListAst=["GA","CO","EP","HE","IM","SR"]
 sourceListAstNames=["Astrophysics of Galaxies","Cosmology and Nongalactic Astrophysics","Earth and Planetary Astrophysics","High Energy Astrophysical Phenomena","Instrumentation and Methods for Astrophysics","Solar and Stellar Astrophysics"]
 SLA = {sourceListAst[i] : sourceListAstNames[i] for i in range(len(sourceListAst))}

 sourceListCon=["dis-nn","mtrl-sci","mes-hall","other","quant-gas","soft","stat-mech","str-el","supr-con"]
 sourceListConNames=["Disordered Systems and Neural Networks","Materials Science","Mesoscale and Nanoscale Physics","Other Condensed Matter","Quantum Gases","Soft Condensed Matter","Statistical Mechanics","Strongly Correlated Electrons","Superconductivity"]
 SLCN = {sourceListCon[i] : sourceListConNames[i] for i in range(len(sourceListCon))}

 sourceListNli=["AO","CG","CD","SI","PS"]
 sourceListNliNames=["Adaptation and Self-Organizing Systems","Cellular Automata and Lattice Gases","Chaotic Dynamics","Exactly Solvable and Integrable Systems","Pattern Formation and Solitons"]
 SLN = {sourceListNli[i] : sourceListNliNames[i] for i in range(len(sourceListNli))}

 sourceListPhy=["acc-ph","ao-ph","atom-ph","atm-clus","bio-ph","chem-ph","class-ph","comp-ph","data-an","flu-dyn","gen-ph","geo-ph","hist-ph","ins-det","med-ph","optics","ed-ph","soc-ph","plasm-ph","pop-ph","space-ph"]
 sourceListPhyNames=["Accelerator Physics","Atmospheric and Oceanic Physics","Atomic Physics","Atomic and Molecular Clusters","Biological Physics","Chemical Physics","Classical Physics","Computational Physics","Data Analysis, Statistics and Probability","Fluid Dynamics","General Physics","Geophysics","History and Philosophy of Physics","Instrumentation and Detectors","Medical Physics","Optics","Physics Education","Physics and Society","Plasma Physics","Popular Physics","Space Physics"]
 SLP = {sourceListPhy[i] : sourceListPhyNames[i] for i in range(len(sourceListPhy))}

 sourceListMat=["AG","AT","AP","CT","CA","CO","AC","CV","DG","DS","FA","GM","GN","GT","GR","HO","IT","KT","LO","MP","MG","NT","NA", "OA","OC","PR","QA","RT","RA","SP","ST","SG"]
 sourceListMatNames=["Algebraic Geometry","Algebraic Topology","Analysis of PDEs","Category Theory","Classical Analysis and ODEs","Combinatorics","Commutative Algebra","Complex Variables","Differential Geometry","Dynamical Systems","Functional Analysis","General Mathematics","General Topology","Geometric Topology","Group Theory","History and Overview","Information Theory","K-Theory and Homology","Logic","Mathematical Physics","Metric Geometry","Number Theory","Numerical Analysis","Operator Algebras","Optimization and Control","Probability","Quantum Algebra","Representation Theory","Rings and Algebras","Spectral Theory","Statistics Theory","Symplectic Geometry"]
 SLM = {sourceListMat[i] : sourceListMatNames[i] for i in range(len(sourceListMat))}

 sourceListCor=["AI","CL","CC","CE","CG","GT","CV","CY","CR","DS","DB","DL","DM","DC","ET","FL","GL","GR","AR","HC","IR", "IT","LG","LO","MS","MA","MM","NI","NE","NA","OS","OH","PF","PL","RO","SI","SE","SD","SC","SY"]
 sourceListCorNames=["Artificial Intelligence","Computation and Language","Computational Complexity","Computational Engineering, Finance, and Science","Computational Geometry","Computer Science and Game Theory","Computer Vision and Pattern Recognition","Computers and Society","Cryptography and Security","Data Structures and Algorithms","Databases","Digital Libraries","Discrete Mathematics","Distributed, Parallel, and Cluster Computing","Emerging Technologies","Formal Languages and Automata Theory","General Literature","Graphics","Hardware Architecture","Human-Computer Interaction","Information Retrieval","Information Theory","Learning","Logic in Computer Science","Mathematical Software","Multiagent Systems","Multimedia","Networking and Internet Architecture","Neural and Evolutionary Computing","Numerical Analysis","Operating Systems","Other Computer Science","Performance","Programming Languages","Robotics","Social and Information Networks","Software Engineering","Sound","Symbolic Computation","Systems and Control"]
 SLCR = {sourceListCor[i] : sourceListCorNames[i] for i in range(len(sourceListCor))}

 sourceListBio=["BM","CB","GN","MN","NC","OT","PE","QM","SC","TO"]
 sourceListBioNames=["Biomolecules","Cell Behavior","Genomics","Molecular Networks","Neurons and Cognition","Other Quantitative Biology","Populations and Evolution","Quantitative Methods","Subcellular Processes","Tissues and Organs"]
 SLB = {sourceListBio[i] : sourceListBioNames[i] for i in range(len(sourceListBio))}

 sourceListFin=["CP","EC","GN","MF","PM","PR","RM","ST","TR"]
 sourceListFinNames=["Computational Finance","Economics","General Finance","Mathematical Finance","Portfolio Management","Pricing of Securities","Risk Management","Statistical Finance","Trading and Market Microstructure"]
 SLF = {sourceListFin[i] : sourceListFinNames[i] for i in range(len(sourceListFin))}

 sourceListSta=["AP","CO","ML","ME","OT","TH"]
 sourceListStaNames=["Applications","Computation","Machine Learning","Methodology","Other Statistics","Statistics Theory"]
 SLS = {sourceListSta[i] : sourceListStaNames[i] for i in range(len(sourceListSta))}

 sourceListMore = [SLA,SLCN,{},{},{},{},{},{},SLN,{},{},SLP,{},SLM,SLCR,SLB,SLF,SLS]	#Would be neater to just use indexes
 idx=[0,1,8,11,131,14,15,16,17]
 SLsub = [[sourceListFullNames[i], sourceListMore[i]] for i in range(len(sourceList))]
 SL = {sourceList[i] : SLsub[i] for i in range(len(sourceList))}	#SL.keys() for keys, (SL.values()[i][1]).keys() for subkeys

 #Don't forget exceptions!
 """i=0	#superkey
 j=1	#subkey
 print str(SL.keys()[i])+"."+str((SL.values()[i][1]).keys()[j])+": "+str(SL.values()[i][0])+", "+str((SL.values()[i][1]).values()[j])"""

 #Bear in mind that none of the below, i.e. physics.class-ph, existed before 2007
 mirrorSelect=0		#uk.arxiv.org/help/mirrors
 mirror=""	#fr	#uk.arxiv.org

 versionSelect=0
 version=""
 versionList=["0.91","1.0","2.0"]

 import time
 date = (time.strftime("%d%m%Y"))
 date = "08052014"	#remove
 choiceIdx=0

 tset = []
 tclass = []
 aclass = []

 i=8
 SL={SL.keys()[i]:SL.values()[i]}	#remove

 for choice in SL.keys():

  #for choice2 in (SL.values()[choiceIdx][1]).keys():

   source = "http://export.arxiv.org/rss/"
   source+=choice

   if mirrorSelect:
	source+="?mirror="+mirror

   if versionSelect:
	vIdx=[version==vItem for vItem in versionList]
	if not(max(vIdx)):
		print "Unacceptable version! Use either 0.91, 1.0, or 2.0\n"
		return
		#sys.exit(1)
	source+="&version="+version

   fromFile=0
   ffile=data_dir+choice+"_data"+date+".pkl"

   if os.path.isfile(ffile)==1:
    if force_get==1:
     os.system("rm "+ffile)
    else:
     fromFile=1

   newdict=[]
   if fromFile==0:
    try:
	llog = feedparser.parse(source)
    except:
	print "Could not grab from arXiv! Please check your mirror.\n"
	return
	#sys.exit(1)

    nentries=len(llog['entries'])
    print "Scraping from "+str(nentries)+" new arXiv "+choice+" entries today"

    newdict = [ llog['entries'][i] for i in range(0,nentries) ]

    with open(ffile,"wb") as ff:
	pickle.dump(newdict,ff,pickle.HIGHEST_PROTOCOL)

   else:

    with open(ffile,"r") as ff:
	newdict = pickle.load(ff)
    nentries = len(newdict)

   #print newdict[0]	#io behaving funny... fix!
   ids = [ newdict[i]['id'] for i in range(0,nentries) ]				#arxiv page
   titles = [ str(newdict[i]['title']) for i in range(0,nentries) ]			#titles
   abstracts = [ str(newdict[i]['summary']) for i in range(0,nentries) ]		#abstract

   #Locate wnd of title
   titleEnd = [ [ fnmatch.fnmatchcase(i,'(arXiv*') for i in titles[i].split() ].index(True) for i in range(0,nentries)]
   idShorts = [ str(ids[i]).replace("http://arxiv.org/abs/","") for i in range(0,nentries) ] 			#stripped out http
   titleShorts = [ " ".join(titles[i].split()[:titleEnd[i]]) for i in range(0,nentries) ]			#stripped out id
   abstractShorts= [ (abstracts[i].replace("<p>","")).replace("</p>","") for i in range(0,nentries) ]		#strip out code

   for i in range(0,nentries):		#could move into the above
	titleShorts[i]=titleShorts[i].lower()
	abstractShorts[i]=abstractShorts[i].lower()

   del ids, titles, abstracts, newdict, titleEnd

   #titleFilt = " ".join([ word.lower() for word in titleShorts[0].split() if len(word)>2 ])
   #print "Filtered for short words\n"

   #Example key words/phrases that pick out what interests me.
   keyweights = [ 1.0/float(i) for i in keyweights ]

   #So now that I have all the data, I need to define precisely what I want to achieve.
   #At the first level, I scan the titles for the keywords. I assign a number for each title describing how many instances of the words I've
   #found.

   if method==0:
    finalList.extend(naiveWeighted(keywords,keyweights,titleShorts,abstractShorts,idShorts,wtThresh,subset=2))

   #Next, we can use something *slightly* more sophisticated. Conditional weights? Bayesian stuff?
   #i.e., compute the probability that a title containing "quantum" and "foundations" is about "quantum foundations"?


   #If I was going to use supervised learning, the approach would be the following:
   #	*Obtain a training set of papers that correspond to the topic I seek
   #	*Clean the titles/abstracts, and get the frequencies of the associated words
   #	*Use those frequencies to classify a new set of titles/abstracts
   #	*We can then combine the topics according to the words I have chosen
   #But since the number of keywords is infinite I will need unsupervised learning: I don't want to predict in advance what keywords a user  
   #wants, grab training sets and go from there --- I don't have the resources for that.

   #Let's try a training set approach anyway. Since I've chosen by titles, I want to get the word frequencies from the abstracts
   elif method==1:

    #get the training set
    if choice==SL.keys()[0]:

     tset = [ getTrainingSet(keywords[2],force_get) ]
     #tset = [ getTrainingSet(key,force_get) for key in keywords ]

    #extract the words from the training set/s
    theWords =  [ get_words_in_text(tseti) for tseti in tset ]

    #For each training set (one per keyword), we classify today's papers 
    for i in range(0,len(tset)):
     #order wordlist by frequency
     word_features = get_word_features(theWords[i])
     #get training set
     training_set = nltk.classify.apply_features(extract_features, tset[i])
     
     classifier=nltk.NaiveBayesClassifier.train(training_set)
     #For each title/abstract, I want to classify it by keyword.
     #Now, I can either do *all* the incoming data, or just the subset that I retrieve from method0. I'll do the former

     #Assuming it works, tclass is a list of whether title i is classified by the keyword key
     tclass = [ classifier.classify(extract_features(titleShorts[i].split())) for i in range(0,nentries) ]
     tclassified.append(tclass)
     #So, I'll have to see whether these are classified correctly. I need a better "other" training set.

     aclass = [ classifier.classify(extract_features(abstractShorts[i].split())) for i in range(0,nentries) ]
     aclassified.append(aclass)	#list containing lists of classifications

    #I now have info for the classes under which each title is classified.
    #For each keyword and weight, I sum with the boolean value for that title/abstract, weighting 2:1 as usual

    #Basically just sum the *true* values for each paper, multiplied by their weights
    i=2
    keywords=keywords[i]	#remove these
    keyweights=keyweights[i]
    tval = [ int(keywords in tclassified[0][j])*keyweights for j in range(nentries) ]
    aval = [ int(keywords in aclassified[0][j])*keyweights for j in range(nentries) ]

    #For each paper j, I return the weighted sum of the number of positive classifications of keyword i (test this)
    #tval = [ sum([ int(keywords[i] in tclassified[i][j])*keyweights[i] for i in range(0,len(keywords)) ]) for j in range(nentries) ]
    #aval = [ sum([ int(keywords[i] in aclassified[i][j])*keyweights[i] for i in range(0,len(keywords)) ]) for j in range(nentries) ]


    for i in range(0,len(tval)):
     if tval[i]>wtThresh:
      finalList.append((idShorts[i],tval[i]))

   #Following Turney, the pointwise mutual information for two words w1 and w2 is: PMI[w1,w2]=Log_2 (p[w1 && w2]/p[w1]p[w2])
   #The semantic orientation is then computed as: SO(phrase) = PMI(phrase,word+)-PMI(phrase,word-)
   #So it works by subtracting the correlations between words; I still need to get the correlation probabilities. But! For my purposes there
   #is no single "not quantum" word that is satisfactory.
   #Where to go from here? Well, I could just measure the absolute correlation with "quantum".


   #A third level would be scanning the main text: but this would be high-bandwidth, slow, and difficult due to pdf/tex issues.
   #It *could* be implemented by the arXiv staff themselves, instead.


   #Unsupered
   #stackoverflow.com/questions/3920759/unsupervised-sentiment-analysis
   #So, using our keywords we wish to classify as "Read" or "Don't read"
  #choiceIdx+=1

 if len(finalList)>1:
   plist = sorted([ [finalList[i][0],finalList[i][1]] for i in range(0,len(finalList)) ],key=getKey,reverse=True)

 if not(finalList):
  print "\nEmpty list of recommendations"
  return

 else:
  print "\nToday's recommended papers are:"
  return(plist)

 #test out scikit as well

