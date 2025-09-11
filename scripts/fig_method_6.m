parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_method_6.png----------

AMPAweights=0.2:0.1:0.8;
score=zeros(length(AMPAweights)+1,5,10); %[AMPA2NMDAratio, R2, In->MN1, MN1->RC, RC->MN1];
for subj=1:10
  disp(subj)
  for j=1:length(AMPAweights)     
      tmp=load(fullfile(parentDir,'fitted_results','bio','fixed_AMPAweight',...
                        sprintf('result_bio_s%d[%g].mat',subj,AMPAweights(j))));
      ref=tmp.ref;
      score(j,:,subj)=[ref.model.AMPAweight,...
                       ref.R2,...
                       mean(ref.model.R),...
                       mean(ref.model.Wexc),...
                       mean(ref.model.RWinh)];
  end
  tmp=load(fullfile(parentDir,'fitted_results','bio',...
                    sprintf('result_bio_s%d.mat',subj)));
  ref=tmp.ref;
  score(end,:,subj)  =[ref.model.AMPAweight,...
                       ref.R2,...
                       mean(ref.model.R),...
                       mean(ref.model.Wexc),...
                       mean(ref.model.RWinh)];
end


figure;
for subj=1:10
 subplot(2,5,subj);
 tmp=sortrows(score(:,:,subj));
  
 h1=plot(tmp(:,1),tmp(:,2),'-ko','linewidth',2);hold on; % R2
 plot(score(end,1,subj),score(end,2,subj),'ro','linewidth',2);
 xlim([0,1]); ylim([0.5 1])
 if ismember(subj,[1,6]),ylabel('R2');end
 yyaxis right
 h2=plot(tmp(:,1),tmp(:,3),'-ko','linewidth',1); % In->MN1
 h3=plot(tmp(:,1),tmp(:,4),'-bo','linewidth',1); % MN1->RC
 h4=plot(tmp(:,1),tmp(:,5),'-mo','linewidth',1); % RC->MN1
 plot(score(end,1,subj),score(end,3,subj),'ro','linewidth',1);
 plot(score(end,1,subj),score(end,4,subj),'ro','linewidth',1);
 plot(score(end,1,subj),score(end,5,subj),'ro','linewidth',1);
 ylim([-1 40])
 title(sprintf('S%d',subj))
 grid on;
 if ismember(subj,8),xlabel('AMPAweight');end
 if ismember(subj,[1,6]),legend([h1,h2,h3,h4],{'R2','R','Wexc','Winh'});end
end

saveas(gcf,['figures' filesep 'fig_method_6.png'])