parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_result_summary.png----------
score=zeros(10,3); %[Pheno,Bio,BioNoRC];
for subj=1:10
 disp(subj)
 tmp=load(fullfile(parentDir,'fitted_results','pheno',sprintf('result_pheno_s%d.mat',subj)));
 [~,ref]=MEPmodel_pheno(tmp.p_post,tmp.ref,0);
 score(subj,1)=ref.R2;
  tmp=load(fullfile(parentDir,'fitted_results','bio',sprintf('result_bio_s%d.mat',subj)));
 [~,ref]=MEPmodel_bio(tmp.p_post,tmp.ref,0);
 score(subj,2)=ref.R2;
  tmp=load(fullfile(parentDir,'fitted_results','bioNoRC',sprintf('result_bioNoRC_s%d.mat',subj)));
 [~,ref]=MEPmodel_bio(tmp.p_post,tmp.ref,0);
 score(subj,3)=ref.R2;
end

figure;
markerColor=[128 128 128]/255;
boxColor=[0,139,139]/255;
width=15;
height=12;
set(gcf,'units','centimeters','position',[2 2 width height])

subplot(2,2,[1 3])
x=repelem(1:3,10)'; y=score(:);
boxchart(x,y);hold on;
xticklabels(['Pheno.','Bio.','Bio.(w/o RC)'])
plot(1:3,score','-o','Color',markerColor);
ylabel('R^2');
box off;ylim([0.5 1])
set(gca,'xticklabel',{'Pheno.','Bio.','Bio.(w/o RC)'},'fontname','calibri')

subplot(222)
scatter(score(:,2),score(:,1),[],markerColor); hold on;
plot([0;1],[0;1],'k');
xlim([0.8 1]);ylim([0.8 1]);
ylabel('R^2 (Pheno.)'); xlabel('R^2 (Bio.)');
for i=1:10
    text(score(i,2)+0.005,score(i,1),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;axis square;


subplot(224)
scatter(score(:,2),score(:,3),[],markerColor); hold on;
plot([0;1],[0;1],'k');
xlim([0.8 1]);ylim([0.5 1]);
ylabel('R^2 (Bio. w/o RC)'); xlabel('R^2 (Bio.)');
for i=1:10
    text(score(i,2)+0.005,score(i,3),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;axis square;
saveas(gcf,['figures' filesep 'fig_result_summary.svg'])

