library(plyr, quietly=TRUE)
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)

Rgb <- read.table("hmnist_8_8_RGB.csv", header=T, sep=",")
meta <- read.table("HAM10000_metadata.csv", header=T, sep=",")


# Combine non-malignant groups and save as y
y <- meta$dx %>%
	revalue(c("akiec"="ab", "bkl"="ben", "df"="ben",
		"bcc"="ab", "nv"="ben", "vasc"="ben")) %>%
	as.factor()
levels(y)
meta$dx <- y
levels(meta$dx)

# ab, define akiec and bcc vs not
ab <- y %>%
	revalue(c("mel"="not",
	 	"ben"="not")) %>%
	as.factor()
levels(ab)

# mel
mel <- y %>%
	revalue(c("ab"="not",
			"ben"="not")) %>%
	as.factor()
levels(mel)

# benign
ben <- y %>%
	revalue(c("ab"="not",
			"mel"="not")) %>%
	as.factor()
levels(ben)

# Make classes data frame

classes <- data.frame(y=y, ab=ab, mel=mel, ben=ben)
head(classes)
summary(classes$y)

# Scale data
print("Scaling...")
Rgb <- as.data.frame(scale(Rgb))

# Expand smaller classes
# print("Expanding...")
# Rgb.exp <- Rgb %>%
#         slice(rep(which(y != "ben"), each=2)) %>%
#         bind_rows(Rgb)
# classes.exp <- classes %>%
#         slice(rep(which(y != "ben"), each=2)) %>%
#         bind_rows(classes)

# Dropping some benign observations
# ben.drop = sample(which(classes.exp$y == "ben"), 1500)
# Rgb.exp <- Rgb.exp[-ben.drop,]
# classes.exp <- classes.exp[-ben.drop,]
#
# dim(classes.exp)
# dim(Rgb.exp)
# summary(classes.exp$y)
# summary(classes.exp$mel)
# summary(classes.exp$ab)
# summary(classes.exp$ben)

# No ben
# Rgb.noben <- Rgb.exp[-which(classes.exp$y == "ben"),]
# classes.noben <- classes.exp[-which(classes.exp$y == "ben"),-4]
# dim(Rgb.noben)
# dim(classes.noben)
# summary(classes.noben)



# write.table(classes, file="classes.txt", sep="\t")
write.csv(classes, file="classes_3cl.csv")

# write.csv(Rgb.exp, file="SER_8_8_RGB.csv")
# write.csv(classes.exp, file="classes_SER_8_8_RGB.csv")

# write.csv(Rgb.exp, file="NOben_8_8_RGB.csv")
# write.csv(classes.exp, file="classes_NOben_8_8_RGB.csv")

