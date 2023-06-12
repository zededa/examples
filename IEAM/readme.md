IBM Edge Application Manager in Zededa platform 


This document describes deploying IBM edge services on the agent running on Zededa platform. 


Agent Based OS: 


1. Ubuntu 18.04 
                   Check this link for an agent installation guide. 
2. RHEL 9.2 


Service dependency diagram:

https://docs.google.com/presentation/d/1TiLNWiNjAAIzjQjMFpC1bmIggbxaDm6UXeMq6y0DspU/edit#slide=id.p
  



 Environment:


 The following environment variables must be set in order to publish services and add deployment policy in IEAM manager.


export HZN_ORG_ID={org-id-YEAM}
export HZN_EXCHANGE_USER_AUTH="iamapikey:{user-auth-token}"
export HZN_EXCHANGE_URL={exchange_url}
export HZN_FSS_CSSURL={CSS_url}

Publish services and add deployment policy to IEAM exchange:

	#export source 
        source env.ibm
	# Publish all services and add deployment
        make publish-all

Update node policy 

	hzn policy update node-policy.json
