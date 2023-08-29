function documents = PreprocessTextF(textData)

% Tokenize the text.
documents = tokenizedDocument(textData);
% Convert to lowercase.
documents = lower(documents);

%{
 Remove a list of stop words then lemmatize the words. To improve
lemmatization, first use addPartOfSpeechDetails.
%}
documents = addPartOfSpeechDetails(documents);
documents = removeStopWords(documents);
documents = normalizeWords(documents,'Style','lemma');

% Erase punctuation.
documents = erasePunctuation(documents);
%{
Remove words with 2 or fewer characters, 
and words with 15 or more characters.
%}
documents = removeShortWords(documents,2);
documents = removeLongWords(documents,15);

end