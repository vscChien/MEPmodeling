function Para_E_new = gradient_repair(Para_E,LR,UR)

if length(LR) == 1 % all parameter use the same constraint
   Para_E(Para_E>UR) = population(1,length(Para_E(Para_E>UR)),LR,UR);
   Para_E(Para_E<LR) = population(1,length(Para_E(Para_E<LR)),LR,UR);
else
   for i = 1:length(Para_E)
        if Para_E(i) > UR(i)
            Para_E(i) = UR(i);
        elseif Para_E(i) < LR(i)
            Para_E(i) = LR(i);
        end  
   end
end

Para_E_new = Para_E;

end